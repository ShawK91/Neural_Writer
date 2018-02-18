import numpy as np, os
import lib_shaw as mod
from random import randint
from torch.autograd import Variable
import torch
from torch.utils import data as util
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F


book_name = "Alice.txt"
save_name = 'alice.pt'
embedding_dim = 1000
max_data = 100000

text = "The analyst"

class Parameters:
    def __init__(self):

        #NN specifics
        self.num_hnodes = 1000
        self.num_mem = 1000

        # Train data
        self.batch_size = 1000
        self.num_epoch = 500
        self.seq_len = 10
        self.prediction_len = 1

        #Dependents
        self.num_output = None
        self.num_input = None
        self.save_foldername = 'R_Word/'
        if not os.path.exists(self.save_foldername): os.makedirs(self.save_foldername)

class Stacked_MMU(nn.Module):
    def __init__(self, input_size, hidden_size, memory_size, output_size, n_vocab):
        super(Stacked_MMU, self).__init__()

        #Define model
        #self.poly = mod.GD_polynet(input_size, hidden_size, hidden_size, hidden_size, None)
        self.embeddings = nn.Embedding(n_vocab+1, embedding_dim)
        self.mmu1 = mod.GD_MMU(embedding_dim, hidden_size, memory_size, hidden_size)
        self.mmu2 = mod.GD_MMU(hidden_size, hidden_size, memory_size, hidden_size)
        self.mmu3 = mod.GD_MMU(hidden_size, hidden_size, memory_size, hidden_size)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.1)


        #self.w_out1 = Parameter(torch.rand(output_size, hidden_size), requires_grad=1)
        #self.w_out2 = Parameter(torch.rand(output_size, hidden_size), requires_grad=1)
        self.w_out3 = Parameter(torch.rand(output_size, hidden_size), requires_grad=1)

        for param in self.parameters():
            # torch.nn.init.xavier_normal(param)
            # torch.nn.init.orthogonal(param)
            # torch.nn.init.sparse(param, sparsity=0.5)
            torch.nn.init.kaiming_normal(param)



    def forward(self, input):
        embeds = self.embeddings(input)
        mmu1_out = self.mmu1.forward(torch.t(embeds))
        mmu1_out = self.dropout1(mmu1_out)
        mmu2_out = self.mmu2.forward(mmu1_out)
        mmu2_out = self.dropout2(mmu2_out)
        mmu3_out = self.mmu3.forward(mmu2_out)
        mmu3_out = self.dropout3(mmu3_out)

        out = self.w_out3.mm(mmu3_out)# + self.w_out2.mm(mmu2_out) + self.w_out1.mm(mmu1_out)
        out  = F.log_softmax(torch.t(out))
        return out





    def reset(self, batch_size):
        #self.poly.reset(batch_size)
        self.mmu1.reset(batch_size)
        self.mmu2.reset(batch_size)
        self.mmu3.reset(batch_size)

class Single_MMU(nn.Module):
    def __init__(self, input_size, hidden_size, memory_size, output_size, n_vocab):
        super(Single_MMU, self).__init__()

        #Define model
        self.embeddings = nn.Embedding(n_vocab+1, embedding_dim)
        self.mmu = mod.GD_MMU(embedding_dim, hidden_size, memory_size, hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.w_out = Parameter(torch.rand(output_size, hidden_size), requires_grad=1)

        for param in self.parameters():
            # torch.nn.init.xavier_normal(param)
            # torch.nn.init.orthogonal(param)
            # torch.nn.init.sparse(param, sparsity=0.5)
            torch.nn.init.kaiming_normal(param)



    def forward(self, input):
        embeds = self.embeddings(input)
        mmu_out = self.mmu.forward(torch.t(embeds))
        mmu_out = self.dropout(mmu_out)
        out = self.w_out.mm(mmu_out)
        out  = F.log_softmax(torch.t(out))
        return out





    def reset(self, batch_size):
        #self.poly.reset(batch_size)
        self.mmu.reset(batch_size)


class Stacked_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, memory_size, output_size, n_vocab):
        super(Stacked_LSTM, self).__init__()

        #Define model
        #self.poly = mod.GD_polynet(input_size, hidden_size, hidden_size, hidden_size, None)
        self.embeddings = nn.Embedding(n_vocab + 1, embedding_dim)
        self.lstm1 = mod.GD_LSTM(embedding_dim, hidden_size, memory_size, hidden_size)
        self.lstm2 = mod.GD_LSTM(hidden_size, hidden_size, memory_size, hidden_size)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)

        #self.w_out1 = Parameter(torch.rand(output_size, hidden_size), requires_grad=1)
        self.w_out2 = Parameter(torch.rand(output_size, hidden_size), requires_grad=1)

        for param in self.parameters():
            # torch.nn.init.xavier_normal(param)
            # torch.nn.init.orthogonal(param)
            # torch.nn.init.sparse(param, sparsity=0.5)
            torch.nn.init.kaiming_normal(param)

    def forward(self, input):
        embeds = self.embeddings(input)
        lstm1_out = self.lstm1.forward(torch.t(embeds))
        lstm1_out = self.dropout1(lstm1_out)
        lstm2_out = self.lstm2.forward(lstm1_out)
        lstm2_out = self.dropout1(lstm2_out)

        out = self.w_out2.mm(lstm2_out)# + self.w_out2.mm(lstm2_out) + self.w_out1.mm(lstm1_out)
        out  = F.log_softmax(torch.t(out))
        return out


    def reset(self, batch_size):
        #self.poly.reset(batch_size)
        self.lstm1.reset(batch_size)
        self.lstm2.reset(batch_size)

class Single_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, memory_size, output_size, n_vocab):
        super(Single_LSTM, self).__init__()

        #Define model
        #self.poly = mod.GD_polynet(input_size, hidden_size, hidden_size, hidden_size, None)
        self.embeddings = nn.Embedding(n_vocab + 1, embedding_dim)
        self.lstm = mod.GD_LSTM(embedding_dim, hidden_size, memory_size, hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.w_out = Parameter(torch.rand(output_size, hidden_size), requires_grad=1)

        for param in self.parameters():
            # torch.nn.init.xavier_normal(param)
            # torch.nn.init.orthogonal(param)
            # torch.nn.init.sparse(param, sparsity=0.5)
            torch.nn.init.kaiming_normal(param)

    def forward(self, input):
        embeds = self.embeddings(input)
        lstm_out = self.lstm.forward(torch.t(embeds))
        lstm_out = self.dropout(lstm_out)

        out = self.w_out.mm(lstm_out)
        out  = F.log_softmax(torch.t(out))
        return out


    def reset(self, batch_size):
        #self.poly.reset(batch_size)
        self.lstm.reset(batch_size)


class Task_Book:
    def __init__(self, parameters):
        self.params = parameters
        self.raw_text, self.char_to_int, self.n_vocab, self.int_to_char = self.read_book()

        #model = Stacked_LSTM(parameters.num_input, parameters.num_hnodes, parameters.num_mem, parameters.num_output, self.n_vocab)
        model = Stacked_MMU(parameters.num_input, parameters.num_hnodes, parameters.num_mem, parameters.num_output, self.n_vocab)
        #model = Single_MMU(parameters.num_input, parameters.num_hnodes, parameters.num_mem, parameters.num_output)
        #model = Single_LSTM(parameters.num_input, parameters.num_hnodes, parameters.num_mem, parameters.num_output)


        self.train_x, self.train_y = self.get_data(seq_len=self.params.seq_len)
        model = torch.load(save_name)
        self.free_write(model)



    def run_bprop(self, model):

        if True: #GD optimizer choices
            #criterion = torch.nn.L1Loss(False)
            #criterion = torch.nn.SmoothL1Loss(False)
            #criterion = torch.nn.KLDivLoss()
            #criterion = torch.nn.CrossEntropyLoss()
            #criterion = torch.nn.MSELoss()
            criterion = torch.nn.NLLLoss()
            #criterion = torch.nn.BCELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            #optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01)
            #optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum = 0.5, nesterov = True)
            #optimizer = torch.optim.RMSprop(model.parameters(), lr = 0.005, momentum=0.1)

        #Get train_data
        seq_len = self.params.seq_len


        #Set up training
        all_train_x = torch.Tensor(self.train_x).long()
        all_train_y = torch.Tensor(self.train_y).long()
        train_dataset = util.TensorDataset(all_train_x, all_train_y)
        train_loader = util.DataLoader(train_dataset, batch_size=self.params.batch_size, shuffle=True)
        model.cuda()
        model.zero_grad()

        for epoch in range(1, self.params.num_epoch):
            epoch_loss = 0.0
            for data in train_loader:  # Each Batch
                net_inputs, targets = data;
                net_inputs = torch.t(net_inputs).cuda(); targets = targets.cuda()
                batch_size = net_inputs.shape[1]
                recall_input = torch.Tensor(torch.zeros((batch_size))+self.n_vocab).long()
                model.reset(batch_size)  # Reset memory and recurrent out for the model

                #Run sequence of chaeacters
                for i in range(seq_len):  # For the length of the sequence
                    net_inp = Variable(net_inputs[i,:], requires_grad=False)
                    model.forward(net_inp)

                #Predict the next character
                for j in range(self.params.prediction_len):
                    net_inp = Variable(recall_input, requires_grad=False).cuda()
                    net_out = model.forward(net_inp)

                    target_T = Variable(targets[:,j])
                    #target_T = torch.max(target_T, 1)[1]
                    loss = criterion(net_out, target_T)
                    loss.backward(retain_variables=True)
                    epoch_loss += loss.cpu().data.numpy()[0]

                optimizer.step()
                optimizer.zero_grad()




            print 'Epoch: ', epoch, ' Loss: ', epoch_loss
            if epoch % 10 == 0: torch.save(model, save_name)

    def read_book(self):

        # load ascii text and covert to lowercase
        raw_text = open(book_name).read()
        raw_text = raw_text.lower().split()

        # create mapping of unique chars to integers
        chars = sorted(list(set(raw_text)))
        char_to_int = dict((c, i) for i, c in enumerate(chars))
        int_to_char = dict((i, c) for i, c in enumerate(chars))

        n_chars = len(raw_text)
        n_vocab = len(chars)
        print "Total Words: ", n_chars
        print "Total Vocab: ", n_vocab

        self.params.num_output = n_vocab
        self.params.num_input = 1

        return raw_text, char_to_int, n_vocab, int_to_char

    def get_data(self, seq_len):

        #Prepare the dataset of input to output pairs encoded as integers
        data_x = []; data_y = []
        for i in range(1):
            seq_in = self.raw_text[i:i + seq_len]
            seq_out = self.raw_text[i + seq_len]
            data_x.append(np.array([self.char_to_int[char] for char in seq_in]))
            data_y.append(np.array([self.char_to_int[seq_out]]))

        n_patterns = len(data_x)


        #Reshape (batch, seq_len, breadth)
        data_x = np.array(data_x); data_y = np.array(data_y)

        return data_x, data_y

    def free_write(self, model):
        net_input = torch.t(torch.Tensor(self.train_x).long().cuda())
        model.cuda()
        model.reset(batch_size=1)
        recall_input = torch.Tensor(torch.zeros((1))+self.n_vocab).long()
        seq_len = self.params.seq_len

        for j in range(1000):
            # Run sequence of characters
            for i in range(seq_len):  # For the length of the sequence
                net_inp = Variable(net_input[i, :], requires_grad=False)
                model.forward(net_inp)

            #Free Write
            net_inp = Variable(recall_input, requires_grad=False).cuda()
            net_out = model.forward(net_inp).cpu().data.numpy()
            print self.int_to_char[int(np.argmax(net_out))],
            if (j+1) % 25 == 0: print

            #Update
            addition = torch.Tensor([int(np.argmax(net_out))]).unsqueeze(0).long().cuda()
            net_input = torch.cat((net_input[1:,:], addition), 0)




if __name__ == "__main__":
    parameters = Parameters()  # Create the Parameters class
    sim_task = Task_Book(parameters)
















