import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
import numpy as np
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
from transformers import AutoConfig
from transformers import AutoModelForTokenClassification

from src.utils import *
from src.dataloader import *
from src.trainer import *
from src.config import *

class BertTagger(nn.Module):
    def __init__(self, hidden_dim, output_dim, model_name):
        super(BertTagger, self).__init__()
        config = AutoConfig.from_pretrained(model_name)
        self.bert_model = AutoModelForTokenClassification.from_pretrained(model_name, config=config)
        self.classifier = nn.Linear(config.hidden_size, output_dim)
    
    def forward(self, X):
        outputs = self.bert_model(X, attention_mask=torch.ones(X.shape).to(X.device), output_hidden_states=True)
        features = outputs.hidden_states[-1]
        logits = self.classifier(features)
        return logits

def main(params):
    if params.seed:
        random.seed(params.seed)
        np.random.seed(params.seed)
        torch.manual_seed(params.seed)
        torch.cuda.manual_seed(params.seed)
        torch.backends.cudnn.deterministic = True
    logger = init_experiment(params, logger_filename=params.logger_filename)
    logger.info(params.__dict__)
    domain_name = os.path.basename(params.data_path[0])
    if domain_name == '':
        domain_name = os.path.basename(params.data_path[0][:-1])
    ner_dataloader = NER_dataloader(data_path=params.data_path,
                                    domain_name=domain_name,
                                    batch_size=params.batch_size, 
                                    entity_list=params.entity_list)
    dataloader_train, dataloader_dev, dataloader_test = ner_dataloader.get_dataloader()
    label_list = ner_dataloader.label_list
    entity_list = ner_dataloader.entity_list

    if params.model_name in ['bert-base-cased', 'roberta-base']:
        model = BertTagger(hidden_dim=params.hidden_dim,
                            output_dim=len(label_list), 
                            model_name=params.model_name)
    else:
        raise Exception('model name %s is invalid' % params.model_name)
    model.cuda()
    trainer = BaseTrainer(params, model, entity_list, label_list)

    logger.info("Training ...")
    no_improvement_num = 0
    best_f1 = 0
    step = 0
    loss_history = []
    f1_history = []

    logger.info("Initial lr is %s" % (str(trainer.scheduler.get_last_lr())))

    for e in range(1, params.training_epochs+1):
        logger.info("============== epoch %d ==============" % e)
        loss_list = []
        mean_loss = 0.0
        total_cnt = 0
        correct_cnt = 0

        pbar = tqdm(dataloader_train, total=len(dataloader_train))
        for X, y in pbar:
            step += 1
            X, y = X.cuda(), y.cuda()
            trainer.batch_forward(X)
            correct_cnt += int(torch.sum(torch.eq(torch.argmax(trainer.logits, dim=2), y).float()).item())
            total_cnt += trainer.logits.size(0) * trainer.logits.size(1)
            trainer.batch_loss(y)
            loss = trainer.batch_backward()
            loss_list.append(loss)
            mean_loss = np.mean(loss_list)
            pbar.set_description("Epoch %d, Step %d: Loss=%.4f, Training_acc=%.2f%%" % (
                e, step, mean_loss, correct_cnt / total_cnt * 100
            ))
        loss_history.append(mean_loss)
        if params.info_per_epochs > 0 and e % params.info_per_epochs == 0:
            logger.info("Epoch %d, Step %d: Loss=%.4f, Training_acc=%.2f%%" % (
                e, step, mean_loss, correct_cnt / total_cnt * 100
            ))
        if trainer.scheduler != None:
            old_lr = trainer.scheduler.get_last_lr()
            trainer.scheduler.step()
            new_lr = trainer.scheduler.get_last_lr()
            if old_lr != new_lr:
                logger.info("Epoch %d, Step %d: lr is %s" % (
                    e, step, str(new_lr)
                ))
        if params.save_per_epochs != 0 and e % params.save_per_epochs == 0:
            trainer.save_model("best_finetune_domain_%s_epoch_%d.pth" % (domain_name, e), path=params.dump_path)
        if e % params.evaluate_interval == 0:
            f1_dev, f1_dev_each_class = trainer.evaluate(dataloader_dev, each_class=True)
            logger.info("Epoch %d, Step %d: Dev_f1=%.4f, Dev_f1_each_class=%s" % (
                e, step, f1_dev, str(f1_dev_each_class)
            ))
            f1_history.append(f1_dev)
            if f1_dev > best_f1:
                logger.info("Find better model!!")
                best_f1 = f1_dev
                no_improvement_num = 0
                trainer.save_model("best_finetune_domain_%s.pth" % domain_name, path=params.dump_path)
            else:
                no_improvement_num += 1
                logger.info("No better model is found (%d/%d)" % (no_improvement_num, params.early_stop))
            if no_improvement_num >= params.early_stop:
                logger.info("Stop training because no better model is found!!!")
                break
    logger.info("Finish training ...")

    logger.info("Testing...")
    trainer.load_model("best_finetune_domain_%s.pth" % domain_name, path=params.dump_path)
    trainer.model.cuda()
    f1_test, f1_score_dict = trainer.evaluate(dataloader_test, each_class=True)
    logger.info("Final Result: Evaluate on Test Set. F1: %.4f." % (f1_test))
    f1_score_dict = sorted(f1_score_dict.items(), key=lambda x: x[0])
    logger.info("F1_list: %s" % (f1_score_dict))
    logger.info("Finish testing ...")

    # Visualize the training process
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(loss_history, label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(f1_history, label='Dev F1 Score')
    plt.title('Dev F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(params.dump_path, 'training_process.png'))
    plt.show()

if __name__ == "__main__":
    params = get_params()
    main(params)