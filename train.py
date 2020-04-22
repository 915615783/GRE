from data_loader.tokenizer import Tokenizer
from data_loader.label_tokenizer import LabelTokenizer
from data_loader.gre_loader import FewRelDataset, get_data_loader

from models.gre import Gre

from torch import optim
import argparse
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_size', default=64, type=int, help='lstm hidden size (encoder and decoder)')
    parser.add_argument('--word_mat_path', default='pretrain/glove/glove_mat.npy', type=str, help='pretrain word embedding')
    parser.add_argument('--num_embeddings', default=400002, type=int)
    parser.add_argument('--embedding_dim', default=50, type=int)

    parser.add_argument('--word2id_path', default='pretrain/glove/glove_word2id.json', type=str)
    parser.add_argument('--id2text_path', default='data/relid2Text.json', type=str)
    parser.add_argument('--trainset_path', default='data/train_wiki.json', type=str)
    parser.add_argument('--valset_path', default='data/val_wiki.json', type=str)
    
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--lr', default=1e-1, type=float)
    parser.add_argument('--epochs', default=20000, type=int)
    parser.add_argument('--val_epochs', default=500, type=int)
    parser.add_argument('--val_per_epochs', default=1000, type=int)
    
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


    # data provide
    # tokenizer
    tokenizer = Tokenizer(word2id_path, is_pad_cut_to_max_length=False, max_length=128)
    label_tokenizer = LabelTokenizer(id2text_path, word2id_path)
    # dataset
    trainset = FewRelDataset(trainset_path, tokenizer, label_tokenizer)
    valset = FewRelDataset(valset_path, tokenizer, label_tokenizer)
    # dataloader
    train_loader = get_data_loader(trainset, batch_size, num_workers=1)
    val_loader = get_data_loader(valset, batch_size, num_workers=1)

    
    # new the model
    model = Gre(hidden_size, num_embeddings, embedding_dim, word_mat_path)
    if torch.cuda.is_available():
        model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 10000, 0.5)


    # training loop
    train_loader_iter = iter(train_loader)
    val_loader_iter = iter(val_loader)
    with tqdm(range(epoch)) as tq:
        for epoch in tq:
            word, pos1, pos2, mask, label_token, label_len = next(train_loader_iter)
            if torch.cuda.is_available():
                





if __name__ == '__main__':
    main()

