import torch
import torch.nn as nn
import torch.nn.functional as F
from model.STPE.STPE import STPE


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        self.values = nn.Linear(embed_size, embed_size, bias=False)
        self.keys = nn.Linear(embed_size, embed_size, bias=False)
        self.queries = nn.Linear(embed_size, embed_size, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        B, K, T, D = x.shape
        # Split the embedding into heads
        values = self.values(x).view(B, K, T, self.heads, self.head_dim)
        keys = self.keys(x).view(B, K, T, self.heads, self.head_dim)
        queries = self.queries(x).view(B, K, T, self.heads, self.head_dim)

        values = values.permute(0, 1, 3, 2, 4)  # B, K, heads, T, head_dim
        keys = keys.permute(0, 1, 3, 2, 4)  # B, K, heads, T, head_dim
        queries = queries.permute(0, 1, 3, 2, 4)  # B, K, heads, T, head_dim

        energy = torch.matmul(queries, keys.transpose(-1, -2))  # B, K, heads, T, head_dim
        attention = F.softmax(energy / (self.embed_size ** (1 / 2)), dim=-1)

        out = torch.matmul(attention, values)  # B, K, heads, T, head_dim
        out = out.permute(0, 1, 3, 2, 4).contiguous()  # N, seq_length, heads, head_dim
        out = out.view(B, K, T, self.embed_size)  # N, seq_length, embed_size

        return self.fc_out(out)


class FeedForward(nn.Module):
    def __init__(self, embed_size, hidden_size):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, embed_size)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, hidden_size, dropout=0):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = FeedForward(embed_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attention = self.attention(x)
        x = self.dropout(self.norm1(attention + x))  # Skip connection
        forward = self.feed_forward(x)
        x = self.dropout(self.norm2(forward + x))  # Skip connection
        return x


class Transformer(nn.Module):
    def __init__(self, embed_size, heads, hidden_size, num_layers, dropout=0):
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList(
            [
                TransformerBlock(embed_size, heads, hidden_size, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class RelationGenerator(nn.Module):
    def __init__(self):
        super().__init__()

    def get_mask(self, s_label, len_q):
        """
            inpput: s_label (shot, t1, 1)
                    len_q (t2)
            output: mask (shot, t2, t1)
        """
        shot = s_label.shape[0]
        len_s = s_label.shape[1]
        mask = s_label.permute(0, 2, 1).repeat(1, len_q, 1)
        assert mask.size() == (shot, len_q, len_s)
        return mask.float()

    def get_similarity(self, query, support, way='cosine', block_size=1000):
        """
            input: query size(i, m, k)
                   support size(j, n, k)
                   way (cosine, euclidean, manhattan)
            output: similarity matrix (i*j, m, n)
        """

        if way == 'cosine':
            i = query.size(0)
            j = support.size(0)
            #  normalize
            query_norm = F.normalize(query, p=2, dim=-1)
            support_norm = F.normalize(support, p=2, dim=-1)
            #  padding
            query_norm = query_norm.unsqueeze(1).expand(-1, support.size(0), -1, -1)  # size(i, j, m, k)
            support_norm = support_norm.unsqueeze(0).expand(query.size(0), -1, -1, -1)  # size(i, j, n, k)
            # calculate similarity
            similarity_matrix = torch.einsum('ijmk,ijnk->ijmn', query_norm, support_norm)
            similarity_matrix = similarity_matrix.view(i, j, query.size(1), support.size(1))
            # S.append(similarity_matrix)
            # shot, t, t
            return similarity_matrix
        elif way == 'euclidean' or 'L2':
            # get size
            i = query.size(0)
            j = support.size(0)
            m = query.size(1)
            n = support.size(1)

            similarity_matrix = torch.zeros(i, j, m, n).to(query.device)

            for i in range(i):
                for j in range(j):
                    l_query = query[i].shape[0]
                    l_support = support[j].shape[0]
                    # 在计算query_expanded 和 support_expanded 的过程当中，继续分块，每block_size长度分一次
                    for query_idx in range(0, l_query, block_size):
                        for support_idx in range(0, l_support, block_size):
                            query_block = query[i][query_idx:query_idx + block_size]
                            support_block = support[j][support_idx:support_idx + block_size]

                            # 扩展维度以便广播
                            query_expanded = query_block.unsqueeze(1)  # size (block_size, 1, k)
                            support_expanded = support_block.unsqueeze(0)  # size (1, block_size, k)
                            # 计算 similarity
                            expanded_matrix = query_expanded - support_expanded
                            similarity_matrix_block = torch.norm(expanded_matrix, p=2, dim=-1)
                            del expanded_matrix
                            torch.cuda.empty_cache()
                            similarity_matrix[i][j][query_idx:query_idx + block_size,
                            support_idx:support_idx + block_size] = similarity_matrix_block

            return similarity_matrix
        elif way == 'manhattan' or 'L1':
            # get size
            i = query.size(0)
            j = support.size(0)
            m = query.size(1)
            n = support.size(1)

            similarity_matrix = torch.zeros(i, j, m, n).to(query.device)

            for i in range(i):
                for j in range(j):
                    l_query = query[i].shape[0]
                    l_support = support[j].shape[0]
                    # 在计算query_expanded 和 support_expanded 的过程当中，继续分块，每block_size长度分一次
                    for query_idx in range(0, l_query, block_size):
                        for support_idx in range(0, l_support, block_size):
                            query_block = query[i][query_idx:query_idx + block_size]
                            support_block = support[j][support_idx:support_idx + block_size]

                            # 扩展维度以便广播
                            query_expanded = query_block.unsqueeze(1)  # size (block_size, 1, k)
                            support_expanded = support_block.unsqueeze(0)  # size (1, block_size, k)
                            # 计算 similarity
                            similarity_matrix_block = torch.norm(query_expanded - support_expanded, p=1, dim=-1)
                            similarity_matrix[i][j][query_idx:query_idx + block_size,
                            support_idx:support_idx + block_size] = similarity_matrix_block
                            del similarity_matrix_block
                            torch.cuda.empty_cache()
            return similarity_matrix

    def forward(self, support, query, s_label=None, q_label=None, mask=False, way='cosine'):

        batch_size = query.shape[0]
        S = []
        M = []

        for b in range(batch_size):
            similarity_matrix = self.get_similarity(query[b], support[b], way=way)
            if mask:
                m = self.get_mask(s_label[b], q_label.size(-2))
                similarity_matrix = torch.mul(similarity_matrix, mask)
                M.append(m)
            S.append(similarity_matrix)
        if mask:
            return torch.stack(S, dim=0), torch.stack(M, dim=0)
        else:
            return torch.stack(S, dim=0)




class Projector(nn.Module):
    def __init__(self, in_channels, out_channels, opt):
        super(Projector, self).__init__()
        self.clip_dim = opt.CLIP_dim
        self.in_channels = in_channels
        self.out_channels = out_channels  # 和in channel 保持一致

        self.Align = nn.Sequential(
            nn.Conv1d(self.in_channels + self.clip_dim, self.in_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(self.in_channels, self.out_channels, kernel_size=1),
        )
        self.q = nn.Linear(self.clip_dim, self.clip_dim)
        self.k = nn.Linear(self.clip_dim, self.clip_dim)
        self.v = nn.Linear(self.clip_dim, self.clip_dim)

    def forward(self, support_video, support_text, support_long_text):
        # all shape of  (batch_size, shot, temporal_scale, feature_dimension)


        mask = torch.any(support_long_text != 0, dim=-1).unsqueeze(-2)

        q = self.q(support_text)
        k = self.k(support_long_text)
        v = self.v(support_long_text)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (q.size(-1) ** 0.5)  # (batch_size, shot, temporal_scale, temporal_scale)
        attn_scores = attn_scores.masked_fill(~mask, -1e9)  # 将填充位置的分数设为-1e9
        attn_weights = F.softmax(attn_scores, dim=-1)
        support_text = torch.matmul(attn_weights, v)

        x = torch.cat([support_video, support_text], dim = -1)
        b, k, t, c = x.shape
        x = x.view(b * k, t, c)
        x = x.transpose(-1, -2)
        x = self.Align(x).transpose(-1, -2)
        x = x.view(b, k, t, -1)
        return x


class Classifier(nn.Module):
    def __init__(self, opt):
        super(Classifier, self).__init__()
        self.opt = opt
        self.in_channels = opt.output_size
        self.Pyraformer = STPE(opt)
        self.GeneralRelationGenerator = RelationGenerator()
        self.SemAlign = Projector(in_channels=opt.output_size, out_channels=opt.output_size, opt=opt)
        self.theta = nn.parameter.Parameter(torch.tensor(0.0, requires_grad=True, device=opt.device))

    def Gaussian(self, input_data, kernel_size=5):
        # Define Gaussian kernels for different sizes
        if kernel_size == 3:
            kernel = torch.tensor([0.2188, 0.5000, 0.2188])  # 3-core Gaussian values
        elif kernel_size == 5:
            kernel = torch.tensor([0.0545, 0.2442, 0.4026, 0.2442, 0.0545])  # 5-core Gaussian values
        elif kernel_size == 7:
            kernel = torch.tensor([0.0039, 0.0350, 0.2419, 0.3876, 0.2419, 0.0350, 0.0039])  # 7-core Gaussian values
        elif kernel_size == 9:
            kernel = torch.tensor(
                [0.0005, 0.0043, 0.0214, 0.0976, 0.2042, 0.0976, 0.0214, 0.0043, 0.0005])  # 9-core Gaussian values
        else:
            raise ValueError("Unsupported kernel size. Use 3, 5, 7, or 9.")

        # Padding the kernel
        kernel = kernel.unsqueeze(0).unsqueeze(0).to(input_data.device)

        # Filter the data
        padding = (kernel_size - 1) // 2
        filter_data = F.conv1d(input_data, kernel, padding=padding).squeeze(0)

        return filter_data.unsqueeze(0)

    def scale_to_0_1_range(self, x):
        x_max = torch.max(x, dim=-1, keepdim=True)[0]
        x_min = torch.min(x, dim=-1, keepdim=True)[0]
        epsilon = torch.tensor(1e-4, device=x.device)
        delta = x_max - x_min
        x = ((x - x_min) + epsilon) / (delta + epsilon)
        return x


    def get_ce(self, probas_q, gt_q):
        """
        inputs:
            probas_q : shape [1, b, 2, temporal_scale, 1]
            gt_q: shape [n_tasks, shot, h, w]

        updates :
             ce : Cross-Entropy between one_hot_gt and probas, shape [n_tasks,]

        """

        # reshape gt-q and probas_q
        probas_q = probas_q[:, :, 0].contiguous().view(-1)
        gt_q = gt_q.view(-1)
        fg_gt = gt_q.float()
        bg_gt = 1 - gt_q.float()
        # limit the probas_q
        probas_q = torch.clamp(probas_q, min=1e-15, max=1 - 1e-15)
        probas_fg = probas_q.squeeze(0)
        probas_bg = 1 - probas_fg

        num_entries = torch.sum(fg_gt, dim=-1) + torch.sum(bg_gt, dim=-1)
        num_positive = torch.sum(fg_gt, dim=-1)
        ratio = (num_entries + 1) / (num_positive + 1)
        # coef_0 = torch.min(0.5 * ratio / (ratio - 1) + 1e-10, torch.tensor([20.0], device=fg_gt.device))
        coef_0 = 0.5 * ratio / (ratio - 1 + 0.001)
        coef_1 = 0.5 * ratio
        coef_0 = coef_0.unsqueeze(-1).repeat(1, probas_fg.size(-1))
        coef_1 = coef_1.unsqueeze(-1).repeat(1, probas_fg.size(-1))
        # criterion
        # criterion = nn.BCELoss(reduction='mean')
        loss_pos = torch.mul(coef_1, torch.log(probas_fg + 1e-10) * fg_gt)
        loss_neg = torch.mul(coef_0, torch.log(probas_bg + 1e-10) * bg_gt)
        ce = -torch.mean(loss_pos + loss_neg)
        # if torch.isnan(ce):
        #     pdb.set_trace()
        return ce

    def TrainEAT(self, support_video, support_text, support_long_text, support_label, query_video, query_text,
                 query_long_text, query_label, lr, iteration,
                 writer=None, episode=None):
        self.train()
        params = list(self.parameters())
        optimizer = torch.optim.Adam(params, lr, weight_decay=1e-5)
        for it in range(1, iteration + 1):
            probas_q = self.forward(support_video, support_text, support_long_text, support_label, query_video,
                                    query_text, query_long_text, query_label,
                                    smooth=False)
            loss = self.get_ce(probas_q, query_label)
            loss.backward()
            optimizer.step()
            writer.add_scalar(f'ce loss episode:{episode + 1}', loss.item(), it)
        return loss

    def forward(self, support_video, support_text, support_long_text, support_label, query_video, query_text,
                query_long_text, query_label, inference=False,
                smooth=False):
        if self.opt.pyraformer:
            support_video = self.Pyraformer(support_video)
            query_video = self.Pyraformer(query_video)
        if self.opt.transformer:
            support_video = self.transformer(support_video)
            query_video = self.transformer(query_video)

        general_relation_map, mask = self.GeneralRelationGenerator(support_video, query_video, support_label,
                                                                   query_label,
                                                                   mask=True)

        support_con = self.SemAlign(support_video, support_text, support_long_text)
        sem_relation_map, mask = self.GeneralRelationGenerator(support_con, query_video, support_label, query_label,
                                                               mask=True)
        final_map = general_relation_map * sem_relation_map


        similarity_vector = torch.max(final_map, dim=-1)[0]
        similarity_vector = torch.mean(similarity_vector, dim=-2)
        if smooth:
            similarity_vector = self.Gaussian(similarity_vector, self.opt.gaussian_kernel)
        similarity_vector = self.scale_to_0_1_range(similarity_vector)
        fg = similarity_vector.unsqueeze(0).unsqueeze(-1)  # torch.Size([1, 3, 1, 256, 1])
        bg = 1 - fg
        proba_q = torch.cat([fg, bg], dim=2)  # (1, b, 2 ,t, 1)
        return proba_q
