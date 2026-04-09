import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class DynamicConvolution(nn.Module):
    def __init__(self, wshare, n_feat, dropout_rate, kernel_size, use_kernel_mask=False, use_bias=False):
        super(DynamicConvolution, self).__init__()
        assert n_feat % wshare == 0
        self.wshare = wshare
        self.use_kernel_mask = use_kernel_mask
        self.dropout_rate = dropout_rate
        self.kernel_size = kernel_size
        self.attn = None

        self.linear1 = nn.Linear(n_feat, n_feat * 2)
        self.linear2 = nn.Linear(n_feat, n_feat)
        self.linear_weight = nn.Linear(n_feat, self.wshare * 1 * kernel_size)
        init.xavier_uniform_(self.linear_weight.weight)
        self.act = nn.GLU()

        self.use_bias = use_bias
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(n_feat))
            self.reset_parameters()

    def reset_parameters(self):
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.)

    def forward(self, query, key, value, count, mask):
        x = query
        B, T, C = x.size()
        H = self.wshare
        k = self.kernel_size
        
        x = self.linear1(x)
        x = self.act(x)
        
        weight = self.linear_weight(x)
        weight = F.dropout(weight, self.dropout_rate, training=self.training)
        weight = weight.view(B, T, H, k).transpose(1, 2).contiguous()
        
        weight_new = torch.zeros(B * H * T * (T + k - 1), dtype=weight.dtype, device=x.device)
        weight_new = weight_new.view(B, H, T, T + k - 1).fill_(float("-inf"))
        
        weight_new.as_strided(
            (B, H, T, k), ((T + k - 1) * T * H, (T + k - 1) * T, T + k, 1)
        ).copy_(weight)
        weight_new = weight_new.narrow(-1, int((k - 1) / 2), T)
        
        if self.use_kernel_mask:
            kernel_mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0)
            weight_new = weight_new.masked_fill(kernel_mask == 0.0, float("-inf"))
        
        weight_new = F.softmax(weight_new, dim=-1)
        self.attn = weight_new
        weight_new = weight_new.view(B * H, T, T)

        x = x.transpose(1, 2).contiguous()
        x = x.view(B * H, int(C / H), T).transpose(1, 2)
        x = torch.bmm(weight_new, x)
        x = x.transpose(1, 2).contiguous().view(B, C, T)

        if self.use_bias:
            x = x + self.bias.view(1, -1, 1)
        x = x.transpose(1, 2)

        if mask is not None and not self.use_kernel_mask:
            mask = mask.transpose(-1, -2)
            x = x.masked_fill(mask == 0, 0.0)

        x = self.linear2(x)
        return x

class MaskedCausalAttention(nn.Module):
    def __init__(self, h_dim, max_T, n_heads, drop_p, kernelsize, dropk, convdim):
        super().__init__()
        self.n_heads = n_heads 
        self.max_T = max_T
        self.drop_p = drop_p
        
        Dimratio = int(convdim)
        self.attdim = h_dim - Dimratio
        self.convdim = Dimratio
        
        if self.attdim != 0:
            self.q_net = nn.Linear(self.attdim, self.attdim)
            self.k_net = nn.Linear(self.attdim, self.attdim)
            self.v_net = nn.Linear(self.attdim, self.attdim)

            self.fmlp = nn.Sequential(
                nn.Linear(h_dim, h_dim // 2),
                nn.GELU(), # 优化：改用 GELU 避免死神经元
            )
            self.fl1 = nn.Linear(h_dim // 2, self.attdim)
            self.fl2 = nn.Linear(h_dim // 2, self.convdim)

        self.proj_net = nn.Linear(h_dim, h_dim)
        self.dropk = dropk
        self.att_drop = nn.Dropout(dropk)
        self.proj_drop = nn.Dropout(drop_p)
        
        ones = torch.ones((max_T, max_T))
        mask = torch.tril(ones).view(1, 1, max_T, max_T)
        self.register_buffer('mask', mask)
        
        self.kernelsize = kernelsize
        self.Dynamicconv_layer = DynamicConvolution(
            wshare=max(1, self.convdim // 4), # 防止 convdim 过小时报错
            n_feat=self.convdim, 
            dropout_rate=0.2, 
            kernel_size=self.kernelsize, 
            use_kernel_mask=True, 
            use_bias=True
        )
        self.count = 0

    def forward(self, x):
        B, T, D = x.shape
        x1, x2 = torch.split(x, [self.attdim, self.convdim], dim=-1)

        if self.attdim != 0:
            attention_future = torch.jit.fork(self.attention_branch, x1, B, T)
            conv_future = torch.jit.fork(self.conv_branch, x2)

            attention_output = torch.jit.wait(attention_future)
            conv_output = torch.jit.wait(conv_future)

            attention2 = torch.cat((attention_output, conv_output), dim=-1)
            
            # Gated Fusion (已修复 Softmax 维度塌陷 Bug)
            Xbar = self.fmlp(attention2)
            X1 = self.fl1(Xbar)
            X2 = self.fl2(Xbar)
            Z = torch.cat((X1, X2), dim=-1)
            
            Xf = torch.sigmoid(Z) # 关键修复：使用 Sigmoid 独立门控
            output = attention2 * Xf # 特征门控调制
        else:
            conv_future = torch.jit.fork(self.conv_branch, x2)
            output = torch.jit.wait(conv_future)

        self.count += 1
        out = self.proj_drop(self.proj_net(output))
        return out
    
    def attention_branch(self, x1, B, T):
        N, D = self.n_heads, x1.size(2) // self.n_heads
        q = self.q_net(x1).view(B, T, N, D).transpose(1, 2) 
        k = self.k_net(x1).view(B, T, N, D).transpose(1, 2)
        v = self.v_net(x1).view(B, T, N, D).transpose(1, 2)

        weights = q @ k.transpose(2, 3) / math.sqrt(D)
        weights = weights.masked_fill(self.mask[..., :T, :T] == 0, float('-inf'))
        weights = F.softmax(weights, dim=-1)
        self.saved_attn_weights = weights.detach()
        attention = self.att_drop(weights @ v)
        attention = attention.transpose(1, 2).contiguous().view(B, T, N * D)
        return attention
    
    def conv_branch(self, x2):
        return self.Dynamicconv_layer(x2, x2, x2, self.count, mask=None)

class Block(nn.Module):
    def __init__(self, h_dim, max_T, n_heads, drop_p, kernelsize, dropk, convdim):
        super().__init__()
        self.attention = MaskedCausalAttention(h_dim, max_T, n_heads, drop_p, kernelsize, dropk, convdim)
        self.mlp = nn.Sequential(
            nn.Linear(h_dim, 4 * h_dim),
            nn.GELU(),
            nn.Linear(4 * h_dim, h_dim),
            nn.Dropout(drop_p),
        )
        self.ln1 = nn.LayerNorm(h_dim)
        self.ln2 = nn.LayerNorm(h_dim)

    def forward(self, x):
        # 关键修复：改为 Pre-LN 架构，彻底解决训练崩溃问题
        x = x + self.attention(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class DecisionTransformer(nn.Module):
    def __init__(self, state_dim, act_dim, n_blocks, h_dim, context_len,
                 n_heads, drop_p, kernelsize, convdim, max_timestep=4096):
        super().__init__()
        self.act_dim = act_dim
        self.h_dim = h_dim

        input_seq_len = 3 * context_len
        blocks = []
        dk = [0.1, 0.1, 0.1, 0.1]
        
        for i in range(n_blocks):
            dropk = dk[i] if i < len(dk) else 0.1
            blocks.append(Block(h_dim, input_seq_len, n_heads, drop_p, kernelsize, dropk, convdim))
            
        self.transformer = nn.Sequential(*blocks)

        self.embed_ln = nn.LayerNorm(h_dim)
        self.embed_timestep = nn.Embedding(max_timestep, h_dim)
        self.embed_rtg = torch.nn.Linear(1, h_dim)
        self.embed_state = torch.nn.Linear(state_dim, h_dim)
        self.embed_action = torch.nn.Linear(act_dim, h_dim)

        self.predict_rtg = torch.nn.Linear(h_dim, 1)
        self.predict_state = torch.nn.Linear(h_dim, state_dim)
        self.predict_action = nn.Sequential(
            nn.Linear(h_dim, act_dim),
            nn.Tanh()
        )
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            init.xavier_uniform_(m.weight)

    def forward(self, timesteps, states, actions, returns_to_go):
        B, T, _ = states.shape
        time_embeddings = self.embed_timestep(timesteps)

        returns_to_go = returns_to_go.float() 
        states = states.float() 
 
        state_embeddings = self.embed_state(states) + time_embeddings
        action_embeddings = self.embed_action(actions) + time_embeddings
        returns_embeddings = self.embed_rtg(returns_to_go) + time_embeddings

        h = torch.stack(
            (returns_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(B, 3 * T, self.h_dim)

        h = self.embed_ln(h)
        h = self.transformer(h)
        h = h.reshape(B, T, 3, self.h_dim).permute(0, 2, 1, 3) 

        return_preds = self.predict_rtg(h[:, 0])     
        state_preds = self.predict_state(h[:, 2])    
        action_preds = self.predict_action(h[:, 1])  

        return state_preds, action_preds, return_preds