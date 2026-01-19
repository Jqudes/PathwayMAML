from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

def evaluate_snaillike(opt, target_files, dataset, model, metatest_tasks=50):
    """
    SNAIL 메타-테스트에서 ROC-AUC, PR-AUC 계산하기
    - opt: argparse.Namespace (필요시 K, cuda 필드 사용)
    - target_files: 메타-테스트용 파일 경로 리스트
    - dataset: GeneExpressionDataset 인스턴스
    - model: 학습된 SnailFewShot
    - metatest_tasks: 각 파일당 반복할 Task 수
    """
    model.eval()
    results = {}
    all_pr, all_auc = [], []

    with torch.no_grad():
        for tgt in target_files:
            # disease 이름 추출 (예: 경로 마지막 폴더명)
            disease = os.path.basename(os.path.dirname(tgt))
            y_true_list, y_prob_list = [], []

            for _ in range(metatest_tasks):
                try:
                    (x_sup, y_sup), (x_q_all, y_q_all) = dataset.create_test_task(opt.K, tgt)
                except ValueError:
                    continue

                # GPU → CPU 순환 없이 바로 모델에 투입
                x_sup = x_sup.to(opt.device)
                y_sup = y_sup.squeeze().long().to(opt.device)

                # Query 전체를 한 번에 처리하는 버전 (앞서 제안한 x_seq_full 방식)
                # (2K + Q, feature_dim) 로 합치고, One-hot 레이블도 통째로 concat
                x_seq_full = torch.cat([x_sup, x_q_all.to(opt.device)], dim=0)
                y_all = torch.cat([y_sup.cpu(), y_q_all.squeeze().cpu()], dim=0)
                y_oh_full = F.one_hot(y_all.long(), num_classes=opt.num_classes).float().to(opt.device)

                output_full = model(x_seq_full, y_oh_full)           # (1, 2K+Q, N)
                logits_q = output_full[:, 2*opt.K:, :].squeeze(0)   # (Q, N)
                probs   = logits_q.softmax(dim=1)[:, 1].cpu().numpy()  # positive class 확률

                # 레이블(0/1) 배열
                y_true = y_q_all.squeeze().long().cpu().numpy()

                y_true_list.extend(y_true.tolist())
                y_prob_list.extend(probs.tolist())

            if len(y_true_list) == 0:
                continue

            # ROC-AUC
            roc = roc_auc_score(y_true_list, y_prob_list)
            # PR-AUC
            prec, rec, _ = precision_recall_curve(y_true_list, y_prob_list)
            pr   = auc(rec, prec)

            results[disease] = (roc, pr)
            all_auc.append(roc)
            all_pr.append(pr)

            print(f"{disease:20s}  ROC-AUC: {roc:.4f}  |  PR-AUC: {pr:.4f}")

    print("-"*50)
    print(f"Overall ROC-AUC: {np.mean(all_auc):.4f}  |  Overall PR-AUC: {np.mean(all_pr):.4f}")
    return results
