import torch
import torch.nn as nn

def xavier_init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv1d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Embedding):
        nn.init.xavier_uniform_(m.weight)

def initialize_weights(model):
    for name, param in model.named_parameters():
        if 'weight' in name:
            if param.dim() > 1:
                if 'conv' in name:
                    nn.init.xavier_uniform_(param)
                elif 'norm' in name:
                    nn.init.constant_(param, 1)
                else:
                    nn.init.xavier_uniform_(param)
            else:
                # 1D 텐서의 경우 uniform 분포로 초기화
                nn.init.uniform_(param, -0.1, 0.1)
        elif 'bias' in name:
            nn.init.constant_(param, 0)

    # BaseModel 특정 초기화
    if hasattr(model, 'base_model'):
        model.base_model.apply(xavier_init_weights)

    # IntensityExtractor 특정 초기화
    if hasattr(model, 'intensity_extractor'):
        if hasattr(model.intensity_extractor, 'emotion_embedding'):
            if hasattr(model.intensity_extractor.emotion_embedding, 'embedding'):
                nn.init.xavier_uniform_(model.intensity_extractor.emotion_embedding.embedding.weight)
        if hasattr(model.intensity_extractor, 'transformer'):
            for layer in model.intensity_extractor.transformer.layers:
                layer.apply(xavier_init_weights)

    # RankModel의 projector 초기화
    # if hasattr(model, 'projector'):
    #     nn.init.xavier_uniform_(model.projector.weight)
    #     nn.init.constant_(model.projector.bias, 0)

    print("Model weights initialized with Xavier initialization (and uniform for 1D tensors).")

