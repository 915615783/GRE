# import sys
# if sys.platform != 'win32':
#     sys.path.pop(-2)

from data_loader.tokenizer import Tokenizer
from data_loader.label_tokenizer import LabelTokenizer
from data_loader.gre_loader import FewRelDataset, get_data_loader
from logger import  logger
from models.gre import Gre

import global_var

import torch
from torch import optim
import argparse
from tqdm import tqdm
import numpy as np
import os

print('pytorch version:', torch.__version__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_size', default=50, type=int, help='lstm hidden size (encoder and decoder)')
    parser.add_argument('--word_mat_path', default='pretrain/glove/glove_mat.npy', type=str, help='pretrain word embedding')
    parser.add_argument('--num_embeddings', default=400002, type=int)
    parser.add_argument('--embedding_dim', default=50, type=int)

    parser.add_argument('--ckpt_load_path', default='./ckpt/bestModel.ckpt', type=str, help='pretrain model path')
    parser.add_argument('--ckpt_save_path', default='./ckpt', type=str)
    
    # parser.add_argument('--ckpt_load_path', default=None, type=str, help='pretrain model path')

    parser.add_argument('--word2id_path', default='pretrain/glove/glove_word2id.json', type=str)
    parser.add_argument('--id2text_path', default='data/relid2Text.json', type=str)
    parser.add_argument('--trainset_path', default='data/train_wiki.json', type=str)
    parser.add_argument('--valset_path', default='data/val_wiki.json', type=str)
    
    parser.add_argument('--batch_size', default=84, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--epochs', default=50000, type=int)
    parser.add_argument('--val_epochs', default=100, type=int)
    parser.add_argument('--val_per_epochs', default=500, type=int)
    parser.add_argument('--is_fix_embedding_and_linear', action='store_true')
    parser.add_argument('--is_teacher_force', action='store_true')
    
    opt = parser.parse_args()

    hidden_size = opt.hidden_size
    word_mat_path = opt.word_mat_path
    num_embeddings = opt.num_embeddings
    embedding_dim = opt.embedding_dim

    word2id_path = opt.word2id_path
    id2text_path = opt.id2text_path
    trainset_path = opt.trainset_path
    valset_path = opt.valset_path

    batch_size = opt.batch_size
    lr = opt.lr
    epochs = opt.epochs
    val_epochs = opt.val_epochs
    val_per_epochs = opt.val_per_epochs

    ckpt_load_path = opt.ckpt_load_path
    ckpt_save_path = opt.ckpt_save_path
    is_fix_embedding_and_linear = opt.is_fix_embedding_and_linear
    is_teacher_force = opt.is_teacher_force

    if not os.path.exists(ckpt_save_path):
        os.mkdir(ckpt_save_path)
        logger.info('Made dir %s' % ckpt_save_path)


    # data provide
    # tokenizer
    tokenizer = Tokenizer(word2id_path, is_pad_cut_to_max_length=False, max_length=128)
    label_tokenizer = LabelTokenizer(id2text_path, word2id_path)
    # assign global_var
    global_var.pad_index = tokenizer.word2id['[PAD]']
    global_var.label_pad_index = label_tokenizer.word2id['[PAD]']
    # dataset
    trainset = FewRelDataset(trainset_path, tokenizer, label_tokenizer)
    valset = FewRelDataset(valset_path, tokenizer, label_tokenizer)
    # dataloader
    train_loader = get_data_loader(trainset, batch_size, num_workers=2)
    val_loader = get_data_loader(valset, batch_size, num_workers=2)

    
    # new the model
    model = Gre(hidden_size, num_embeddings, embedding_dim, word_mat_path, label_pad_index=label_tokenizer.word2id['[PAD]'], is_teacher_force=is_teacher_force)
    # # use word2vec^{-1} to init the output layer
    # logger.info('Using word2vec^{-1} to init the output layer')
    # with torch.no_grad():
    #     model.decoder.linear.weight[:-1, :] = torch.tensor(np.linalg.pinv(model.encoder.embedding.weight.detach().numpy().T), dtype=torch.float32)

    # load ckpt
    if (ckpt_load_path != None) and (os.path.isfile(ckpt_load_path)):
        logger.info('Loading pretrain model from %s ...' % ckpt_load_path)
        pretrain_state_dict = torch.load(ckpt_load_path)
        model_state_dict = model.state_dict()
        # filter the variable if model changed
        new_dict = {}
        for k in pretrain_state_dict.keys():
            if k in model_state_dict:
                new_dict[k] = pretrain_state_dict[k]
        model_state_dict.update(new_dict)
        model.load_state_dict(model_state_dict)
        logger.info('Loaded. Total pretrain variables: %d, Update: %d' %(len(pretrain_state_dict), len(new_dict)))

    # fix some layer
    if is_fix_embedding_and_linear:
        fix_var = []
        fix_var.extend(model.encoder.embedding.parameters())
        fix_var.extend(model.decoder.embedding.parameters())
        fix_var.extend(model.decoder.linear.parameters())
        for i in fix_var:
            i.requires_grad = False
        logger.info('Fixed Embedding and the output linear')
    
    

    if torch.cuda.is_available():
        logger.info('Using cuda.')
        model.cuda()

    optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 10000, 0.5)

    # training loop
    train_loader_iter = iter(train_loader)
    latest_losses = []
    average_loss = None
    average_acc = None
    best_metric_to_save = -100000
    with tqdm(range(epochs)) as tq:
        for epoch in tq:
            model.train()
            word, pos1, pos2, mask, label_token, label_len = next(train_loader_iter)
            # move to gpu if available
            if torch.cuda.is_available():
                word = word.cuda()
                pos1 = pos1.cuda()
                pos2 = pos2.cuda()
                mask = mask.cuda()
                label_token = label_token.cuda()
                label_len = label_len.cuda()
            
            out, loss = model(word, pos1, pos2, mask, label_token, label_len)

            if np.random.randn() < -1.8:
                # print(decoder_out[0:3].topk(1, dim=-1)[1], label_token[0:3])
                logger.info(' ')
                for i in range(0, 81, 15):
                    logger.info('P '+str([label_tokenizer.id2word[index] if index!=len(label_tokenizer.id2word) else '[EOS]' for index in out[i].topk(1, dim=-1)[1].cpu().numpy().reshape(-1).tolist()]))
                    logger.info('L '+str([label_tokenizer.id2word[index] if index!=len(label_tokenizer.id2word) else '[EOS]' for index in label_token[i].cpu().numpy().reshape(-1).tolist()]))

            if average_acc == None:
                average_acc = accuracy(out, label_token, label_tokenizer)
            else:
                average_acc = average_acc*0.95 + accuracy(out, label_token, label_tokenizer)*0.05

            # 两种统计loss
            if average_loss == None:
                average_loss = loss.item()
            else:
                average_loss = average_loss*0.95 + (loss.item())*0.05
            latest_losses.append(loss.item())
            if len(latest_losses) >100:
                latest_losses.pop(0)
            tq.set_description('loss: %f, average_loss: %f, average_acc: %f' % (np.mean(latest_losses), average_loss, average_acc))
            loss.backward()

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            # evaluation
            if (epoch % val_per_epochs == 0) and (epoch != 0) :
                eval_loss, eval_acc = eval(model, val_loader, val_epochs, label_tokenizer)
                # save_ckpt
                if ckpt_save_path != None:
                    if (eval_acc > best_metric_to_save) or True:   # 这里临时加的，无论如何都保存ckpt
                        best_metric_to_save = eval_acc
                        logger.info('Best Ckpt Saved!')
                        state_dict = model.state_dict()
                        torch.save(state_dict, ckpt_save_path + '/bestModelfix.ckpt')

def accuracy(out, label_token, label_tokenizer):
    with torch.no_grad():
        out = out.topk(1, dim=-1)[1].squeeze(-1).cpu().numpy()  # (b, max_l)
        label = label_token.cpu().numpy()  # (b, max_l)
        mask = (label != label_tokenizer.word2id['[PAD]']).astype(np.int)
        return (((out == label).astype(np.int) * mask).sum())/(mask.sum()) 


def eval(model, val_loader, val_epochs, label_tokenizer):
    '''
    return ave_loss, ave_acc
    '''
    logger.info('Begin evaluation:=============================================')
    val_loader_iter = iter(val_loader)
    latest_losses = []
    acces = []
    for epoch in range(val_epochs):
        model.eval()
        with torch.no_grad():
            word, pos1, pos2, mask, label_token, label_len = next(val_loader_iter)
            # move to gpu if available
            if torch.cuda.is_available():
                word = word.cuda()
                pos1 = pos1.cuda()
                pos2 = pos2.cuda()
                mask = mask.cuda()
                label_token = label_token.cuda()
                label_len = label_len.cuda()
            
            out, loss = model(word, pos1, pos2, mask, label_token, label_len)
            if np.random.randn() < -1.8:
                # print(decoder_out[0:3].topk(1, dim=-1)[1], label_token[0:3])
                for i in range(0, 81, 8):
                    logger.info(' ')
                    logger.info('P '+str([label_tokenizer.id2word[index] if index!=len(label_tokenizer.id2word) else '[EOS]' for index in out[i].topk(1, dim=-1)[1].cpu().numpy().reshape(-1).tolist()]))
                    logger.info('L '+str([label_tokenizer.id2word[index] if index!=len(label_tokenizer.id2word) else '[EOS]' for index in label_token[i].cpu().numpy().reshape(-1).tolist()]))
            
            latest_losses.append(loss.item())
            acces.append(accuracy(out, label_token, label_tokenizer))
    logger.info('Eval loss: %f, acc: %f' % (np.mean(latest_losses), np.mean(acces)))
    logger.info('END evaluation:  =========================================================')
    return np.mean(latest_losses), np.mean(acces)


            

            


            
                




if __name__ == '__main__':
    main()

