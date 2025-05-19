import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset33 import EmotionalSpeechDataset, collate_fn
from models.model_enhance import RankModel
from xavier_initialize import initialize_weights
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

import time
from datetime import timedelta
from loss import *
import yaml
import os


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def split_data(samples, train_ratio=0.9, val_ratio=0.05, test_ratio=0.05):
    train_samples, temp_samples = train_test_split(samples, train_size=train_ratio, random_state=42)
    val_samples, test_samples = train_test_split(temp_samples, train_size=val_ratio / (val_ratio + test_ratio),
                                                 random_state=42)
    return train_samples, val_samples, test_samples


def get_data_loaders(config):
    preprocessed_path = config['preprocessing']['preprocessed_path']
    speakers = config['preprocessing']['speakers']
    txt_paths = {speaker: os.path.join(preprocessed_path, f"{speaker}.txt") for speaker in speakers}



    all_pairs = []
    for speaker in speakers:
        dataset = EmotionalSpeechDataset(preprocessed_path, {speaker: txt_paths[speaker]})
        all_pairs.extend(dataset.pairs)

    train_pairs, val_pairs, test_pairs = split_data(all_pairs)

    train_dataset = EmotionalSpeechDataset(preprocessed_path, txt_paths)
    train_dataset.pairs = train_pairs

    val_dataset = EmotionalSpeechDataset(preprocessed_path, txt_paths)
    val_dataset.pairs = val_pairs

    test_dataset = EmotionalSpeechDataset(preprocessed_path, txt_paths)
    test_dataset.pairs = test_pairs

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False
    )

    return train_loader, val_loader, test_loader


import time
from datetime import timedelta


def train(config, model, train_loader, val_loader, device):

    optimizer = optim.Adam(model.parameters(), lr=float(config['training']['learning_rate']))
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=25, verbose=True)

    writer = SummaryWriter(log_dir=config['training']['log_dir'])

    step = 0
    best_val_loss = float('inf')

    total_epochs = config['training']['max_steps'] // len(train_loader) + 1
    epoch_times = []

    start_time = time.time()
    epoch_iterator = trange(total_epochs, desc="Training", unit="epoch")
    for epoch in epoch_iterator:
        epoch_start_time = time.time()
        model.train()
        current_lr = optimizer.param_groups[0]['lr']

        train_iterator = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{total_epochs}", leave=False)
        for batch in train_iterator:
            if batch==None:
                continue
            if step >= config['training']['max_steps']:
                break

            x_emo, x_neu, emotion_class, y_emo, y_neu, lengths_emo, lengths_neu = [b.to(device) for b in batch]

            optimizer.zero_grad()

            h_mix_i, h_mix_j, r_mix_i, r_mix_j, lambda_i, lambda_j = model(x_emo, x_neu, emotion_class)

            l_mixup, l_rank, loss = total_loss(h_mix_i, h_mix_j, r_mix_i, r_mix_j, y_emo, y_neu, lambda_i, lambda_j,
                              config['rank_model']['alpha'], config['rank_model']['beta'])


            l_mixup, l_rank = l_mixup.mean(), l_rank.mean()

            loss.backward()
            optimizer.step()

            writer.add_scalar('train_loss/total', loss.item(), step)
            writer.add_scalar('train_loss/mixup', l_mixup.item(), step)
            writer.add_scalar('train_loss/rank', l_rank.item(), step)

            train_iterator.set_postfix({"loss": f"{loss.item():.4f}", "mixup_loss": f"{l_mixup.item():.4f}", "rank_loss": f"{l_rank.item():.4f}", "LR": f"{current_lr}"})
            # train_iterator.set_postfix({"mixup_loss": f"{l_mixup.item():.4f}"})
            # train_iterator.set_postfix({"rank_loss": f"{l_rank.item():.4f}"})

            if step % config['training']['eval_step'] == 0:
                mix_val_loss, rank_val_loss, val_loss = validate(model, val_loader, config, device, writer, step)
                epoch_iterator.write(f"Step {step}/{config['training']['max_steps']}")
                epoch_iterator.write(f"Train Loss: {loss.item():.4f}")
                epoch_iterator.write(f"Val Loss: {val_loss:.4f}")
                scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), config['training']['model_save_path'])
                    epoch_iterator.write(f"Best model saved with validation loss: {best_val_loss:.4f}")

            if step % config['training']['save_step'] == 0:
                torch.save(model.state_dict(), f"/home/lee/Desktop/INSUNG/intensity_fs2/tts_tool/intensity_train/runs/inter_intra/model_step_{step}.pth")

            step += 1

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_duration)
        avg_epoch_time = sum(epoch_times) / len(epoch_times)

        remaining_epochs = total_epochs - (epoch + 1)
        estimated_remaining_time = remaining_epochs * avg_epoch_time

        epoch_iterator.set_postfix({
            "epoch_time": str(timedelta(seconds=int(epoch_duration))),
            "remaining_time": str(timedelta(seconds=int(estimated_remaining_time)))
        })

        total_elapsed_time = time.time() - start_time
        epoch_iterator.write(f"Total training time so far: {timedelta(seconds=int(total_elapsed_time))}")

    writer.close()

    total_training_time = time.time() - start_time
    print(f"Total training time: {timedelta(seconds=int(total_training_time))}")


def validate(model, val_loader, config, device, writer, step):
    model.eval()
    total_losses = 0
    total_mixup_losses = 0
    total_rank_losses = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation", leave=False):
            if batch==None:
                continue
            x_emo, x_neu, emotion_class, y_emo, y_neu, _, _ = [b.to(device) for b in batch]
            h_mix_i, h_mix_j, r_mix_i, r_mix_j, lambda_i, lambda_j = model(x_emo, x_neu, emotion_class)
            l_mixup, l_rank, loss = total_loss(h_mix_i, h_mix_j, r_mix_i, r_mix_j, y_emo, y_neu, lambda_i, lambda_j,
                              config['rank_model']['alpha'], config['rank_model']['beta'])

            l_mixup, l_rank = l_mixup.mean(), l_rank.mean()
            total_losses += loss.item()
            total_mixup_losses += l_mixup.item()
            total_rank_losses += l_rank.item()

    avg_val_loss = total_losses / len(val_loader)
    avg_val_l_mixup = total_mixup_losses / len(val_loader)
    avg_val_l_rank = total_rank_losses / len(val_loader)

    writer.add_scalar('val_loss/total', avg_val_loss, step)
    writer.add_scalar('val_loss/mixup', avg_val_l_mixup, step)
    writer.add_scalar('val_loss/rank', avg_val_l_rank, step)


    return avg_val_l_mixup, avg_val_l_rank, avg_val_loss


def test_emotion_intensity(model, test_loader, device):
    model.eval()
    results = []
    correct_count = 0
    wrong_count = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            if batch==None:
                continue
            x_emo, x_neu, emotion_class, y_emo, y_neu,_, _ = [b.to(device) for b in batch]

            h_mix_i, h_mix_j, r_mix_i, r_mix_j, lambda_i, lambda_j = model(x_emo, x_neu, emotion_class)

            # todo : lambda i가 lambda j보다 클때 또는 작을때 r_mix_i, r_mix_j도 lambda의 차이를 따라서 크거나 작아야 한다

            for i in range(lambda_i.size(0)):
                if lambda_i[i] > lambda_j[i]:
                    if r_mix_i[i] > r_mix_j[i]:
                        correct_count += 1
                    else:
                        wrong_count += 1
                else:
                    if r_mix_i[i] < r_mix_j[i]:
                        correct_count += 1
                    else:
                        wrong_count += 1

            # 결과 저장
            for i in range(x_emo.size(0)):
                results.append({
                    'emotion': emotion_class[i].item(),
                    'score_i': r_mix_i[i].item(),
                    'score_j': r_mix_j[i].item(),
                    'lambda_i': lambda_i[i].item(),
                    'lambda_j': lambda_j[i].item(),
                    'true_emotion': y_emo[i].item(),
                    'neutral_emotion': y_neu[i].item()
                })
    total_count = wrong_count + correct_count
    correct = correct_count / total_count
    wrong = wrong_count / total_count
    return results, correct, wrong


if __name__ == "__main__":
    config = load_config('./config/emo_intensity/intensity.yaml')

    # device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    device = 'cpu'
    train_loader, val_loader, test_loader = get_data_loaders(config)
    model_config = config['model']
    model = RankModel(model_config).to(device)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    initialize_weights(model)

    train(config, model, train_loader, val_loader, device)

    model.load_state_dict(torch.load(config['training']['model_save_path']))

    # 테스트 실행
    test_results, correct, wrong = test_emotion_intensity(model, test_loader, device)

    # 결과 분석 및 출력
    for result in test_results[:10]:  # 처음 10개 결과만 출력
        print(f"Emotion: {result['emotion']}, Strong Score: {result['score_i']:.4f}, "
              f"Weak Score: {result['score_j']:.4f}, Lambda i: {result['lambda_i']:.4f}, "
              f"Lambda j: {result['lambda_j']:.4f}, True Emotion: {result['true_emotion']}, "
              f"Neutral Emotion: {result['neutral_emotion']}")

    # 통계 계산
    avg_strong_score = sum(r['score_i'] for r in test_results) / len(test_results)
    avg_weak_score = sum(r['score_j'] for r in test_results) / len(test_results)
    avg_lambda_i = sum(r['lambda_i'] for r in test_results) / len(test_results)
    avg_lambda_j = sum(r['lambda_j'] for r in test_results) / len(test_results)
    print(f"\nAverage Strong Score: {avg_strong_score:.4f}")
    print(f"Average Weak Score: {avg_weak_score:.4f}")
    print(f"Average Lambda i: {avg_lambda_i:.4f}")
    print(f"Average Lambda j: {avg_lambda_j:.4f}")
    print('correct:', correct)
    print('wrong:', wrong)