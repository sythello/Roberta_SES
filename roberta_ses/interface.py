import torch
from torch.nn import functional as F
import pytorch_lightning as pl
from transformers import RobertaTokenizer

from roberta_ses.datasets.collate_functions import collate_to_max_length
from roberta_ses.explain.model import ExplainableModel

class Roberta_SES_Entailment(pl.LightningModule):
    def __init__(self,
                 roberta_path,
                 ckpt_path,
                 output_classes=None,
                 max_length=512,
                 device_name='cpu'):
        super().__init__()
        
        self.roberta_path = roberta_path
        self.ckpt_path = ckpt_path
        self.output_classes = output_classes
        self.max_length = max_length
        self.device_name = device_name
        
        # For loading purpose, this has to be named "model" 
        self.model = ExplainableModel(roberta_path, output_classes)
        
        self.tokenizer = RobertaTokenizer.from_pretrained(roberta_path)
        
        # Loading trained weights 
        checkpoint = torch.load(ckpt_path, map_location=torch.device(device_name))
        self.load_state_dict(checkpoint['state_dict'])

    def sent_to_tensor(self, sentence_1, sentence_2):
        # remove .
        if sentence_1.endswith("."):
            sentence_1 = sentence_1[:-1]
        if sentence_2.endswith("."):
            sentence_2 = sentence_2[:-1]
        sentence_1_input_ids = self.tokenizer.encode(sentence_1, add_special_tokens=False)
        sentence_2_input_ids = self.tokenizer.encode(sentence_2, add_special_tokens=False)
        input_ids = sentence_1_input_ids + [2] + sentence_2_input_ids
        if len(input_ids) > self.max_length - 2:
            input_ids = input_ids[:self.max_length - 2]

        # convert list to tensor
        length = torch.LongTensor([len(input_ids) + 2])
        input_ids = torch.LongTensor([0] + input_ids + [2])
        label = torch.LongTensor([-1]) ## Simply placeholder 
#         label = None
        return input_ids, label, length

    def compute_loss_and_acc(self, batch):
        input_ids, labels, length, start_indexs, end_indexs, span_masks = batch
    #     y = labels.view(-1)
        with torch.no_grad():
            y_hat, a_ij = self.model(input_ids, start_indexs, end_indexs, span_masks)
        # compute loss
    #     ce_loss = self.loss_fn(y_hat, y)
    #     reg_loss = self.args.lamb * a_ij.pow(2).sum(dim=1).mean()
    #     loss = ce_loss - reg_loss
    
        predict_scores = F.softmax(y_hat, dim=1)
        predict_labels = torch.argmax(predict_scores, dim=-1)

        return predict_labels, predict_scores

    def predict(self, sentence_1, sentence_2):
        # 0 = contra, 1 = neutral, 2 = entail 
        input_ids, label, length = self.sent_to_tensor(sentence_1, sentence_2)
        batch = [[input_ids, label, length]]
        batch = collate_to_max_length(batch)
        predict_labels, predict_scores = self.compute_loss_and_acc(batch)
        pred_label = predict_labels[0]
        pred_score = predict_scores[0]
        return pred_label, pred_score



class Roberta_SES_Sentiment(pl.LightningModule):
    def __init__(self,
                 roberta_path,
                 ckpt_path,
                 output_classes=None,
                 max_length=512,
                 device_name='cpu'):
        super().__init__()
        
        self.roberta_path = roberta_path
        self.ckpt_path = ckpt_path
        self.output_classes = output_classes
        self.max_length = max_length
        self.device_name = device_name
        
        # For loading purpose, this has to be named "model" 
        self.model = ExplainableModel(roberta_path, output_classes)
        
        self.tokenizer = RobertaTokenizer.from_pretrained(roberta_path)
        
        # Loading trained weights 
        checkpoint = torch.load(ckpt_path, map_location=torch.device(device_name))
        self.load_state_dict(checkpoint['state_dict'])

    def sent_to_tensor(self, sentence_1):
        # remove .
        if sentence_1.endswith("."):
            sentence_1 = sentence_1[:-1]

        sentence_1_input_ids = self.tokenizer.encode(sentence_1, add_special_tokens=False)

        input_ids = sentence_1_input_ids
        if len(input_ids) > self.max_length - 2:
            input_ids = input_ids[:self.max_length - 2]

        # convert list to tensor
        length = torch.LongTensor([len(input_ids) + 2])
        input_ids = torch.LongTensor([0] + input_ids + [2])
        label = torch.LongTensor([-1]) ## Simply placeholder 
        return input_ids, label, length

    def compute_loss_and_acc(self, batch):
        input_ids, labels, length, start_indexs, end_indexs, span_masks = batch
    #     y = labels.view(-1)
        with torch.no_grad():
            y_hat, a_ij = self.model(input_ids, start_indexs, end_indexs, span_masks)
        # compute loss
    #     ce_loss = self.loss_fn(y_hat, y)
    #     reg_loss = self.args.lamb * a_ij.pow(2).sum(dim=1).mean()
    #     loss = ce_loss - reg_loss
    
        predict_scores = F.softmax(y_hat, dim=1)
        predict_labels = torch.argmax(predict_scores, dim=-1)

        return predict_labels, predict_scores

    def predict(self, sentence_1):
        # 0 -> 4: neg -> pos
        input_ids, label, length = self.sent_to_tensor(sentence_1)
        batch = [[input_ids, label, length]]
        batch = collate_to_max_length(batch)
        predict_labels, predict_scores = self.compute_loss_and_acc(batch)
        pred_label = predict_labels[0]
        pred_score = predict_scores[0]
        return pred_label, pred_score


if __name__ == "__main__":
    # roberta_path = '/home/ubuntu/users/yutong/models/roberta-base'
    roberta_path = '/home/ec2-user/ytshao/models/roberta-base'

    ## entailment test
    print("Creating entailment object...")
    entailment_model = Roberta_SES_Entailment(roberta_path=roberta_path,
                                              ckpt_path='checkpoints/snli_checkpoints/base/epoch=4-valid_loss=-0.6472-valid_acc_end=0.9173.ckpt',
                                              max_length=512,
                                              output_classes=3,
                                              device_name='cpu')
    print("done.")
    print()
    print("Predicting...")
    sentence_1 = "In walmart : no, you get paid biweekly"
    sentence_2 = "walmart pays biweekly"
    pred = entailment_model.predict(sentence_1, sentence_2)
    print(sentence_1)
    print(sentence_2)
    print(f"Prediction: {pred}")
    print()

    ## sentiment test
    print("Creating sentiment object...")
    sentiment_model = Roberta_SES_Sentiment(roberta_path=roberta_path,
                                              ckpt_path='checkpoints/sst5_checkpoints/base/epoch=2-valid_loss=0.9624-valid_acc_end=0.5746.ckpt',
                                              max_length=512,
                                              output_classes=5,
                                              device_name='cpu')
    print("done.")
    print()
    print("Predicting...")
    sent = "This cream is not only ultra hydrating, it absorbs quickly into the skin. "
    pred = sentiment_model.predict(sent)
    print(sent)
    print(f"Prediction: {pred}")






