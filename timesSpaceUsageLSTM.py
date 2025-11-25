# %%
import torch
import time

# %% [markdown]
# ## Delayed XOR test

# %%
# from custom_datasets.XOR_dataset import Delayed_XOR_DataSet as Dataset

# for t in range(10):
#     lstm1 = torch.nn.LSTM(2, 3)
#     lstm2 = torch.nn.LSTM(3, 1)
#     model = torch.nn.ModuleList([lstm1, lstm2])
#     dataset = Dataset()
#     optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
#     mse = torch.nn.MSELoss()
#     print('starting trail', t)
#     for e in range(3000):
#         dataset.shuffle()
#         err_num = 0
#         for raw_xs, raw_y in dataset:
#             hidden1 = (torch.zeros(1, 1, 3, dtype=torch.float32),
#                     torch.zeros(1, 1, 3, dtype=torch.float32))
#             hidden2 = (torch.zeros(1, 1, 1, dtype=torch.float32),
#                     torch.zeros(1, 1, 1, dtype=torch.float32))
#             y = torch.tensor(raw_y[0], dtype=hidden1[0].dtype)
#             for raw_x in raw_xs:
#                 x = torch.tensor(raw_x, dtype=torch.float32).view(1,1,-1)
#                 out1, hidden1 = lstm1(x, hidden1)
#                 out2, hidden2 = lstm2(out1, hidden2)

#             if (out2 > 0.5 and raw_y[0] < 1) or (out2 < 0.5 and raw_y[0] > 0):
#                 err_num = err_num + 1
#                 loss = mse(out2, y)
#                 loss.backward()
#                 optimizer.step()
#                 # print(loss)
#                 # print('y_hat', out2)
#                 # print('raw_y', raw_y)
#         if err_num == 0:
#             print('reach full acc at epch', e, '\n')
#             break
#         # print(err_num)
#     if err_num != 0:
#         print('fail to reach full acc', '\n')


# %% [markdown]
# ## single channel XOR

# %%
# from custom_datasets.XOR_dataset import Single_Channel_XOR_DataSet_with_invert as Dataset

# for t in range(10):
#     lstm1 = torch.nn.LSTM(1, 3)
#     lstm2 = torch.nn.LSTM(3, 1)
#     model = torch.nn.ModuleList([lstm1, lstm2])
#     dataset = Dataset()
#     optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
#     mse = torch.nn.MSELoss()
#     print('starting trail', t)
#     for e in range(3000):
#         dataset.shuffle()
#         err_num = 0
#         for raw_xs, raw_y in dataset:
#             hidden1 = (torch.zeros(1, 1, 3, dtype=torch.float32),
#                     torch.zeros(1, 1, 3, dtype=torch.float32))
#             hidden2 = (torch.zeros(1, 1, 1, dtype=torch.float32),
#                     torch.zeros(1, 1, 1, dtype=torch.float32))
#             y = torch.tensor(raw_y[0], dtype=hidden1[0].dtype)
#             for raw_x in raw_xs:
#                 x = torch.tensor(raw_x, dtype=torch.float32).view(1,1,-1)
#                 out1, hidden1 = lstm1(x, hidden1)
#                 out2, hidden2 = lstm2(out1, hidden2)

#             if (out2 > 0.5 and raw_y[0] < 1) or (out2 < 0.5 and raw_y[0] > 0):
#                 err_num = err_num + 1
#                 loss = mse(out2, y)
#                 loss.backward()
#                 optimizer.step()
#                 # print(loss)
#                 # print('y_hat', out2)
#                 # print('raw_y', raw_y)
#         if err_num == 0:
#             print('reach full acc at epch', e, '\n')
#             break
#         # print(err_num)
#     if err_num != 0:
#         print('fail to reach full acc', '\n')


# %% [markdown]
# ## single channel with interruption

# %%
# from custom_datasets.XOR_dataset import Single_Channel_XOR_DataSet_with_invert_with_interruption as Dataset

# for t in range(10):
#     lstm1 = torch.nn.LSTM(1, 3)
#     lstm2 = torch.nn.LSTM(3, 1)
#     model = torch.nn.ModuleList([lstm1, lstm2])
#     dataset = Dataset()
#     optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
#     mse = torch.nn.MSELoss()
#     print('starting trail', t)
#     for e in range(3000):
#         dataset.shuffle()
#         err_num = 0
#         for raw_xs, raw_y in dataset:
#             hidden1 = (torch.zeros(1, 1, 3, dtype=torch.float32),
#                     torch.zeros(1, 1, 3, dtype=torch.float32))
#             hidden2 = (torch.zeros(1, 1, 1, dtype=torch.float32),
#                     torch.zeros(1, 1, 1, dtype=torch.float32))
#             y = torch.tensor(raw_y[0], dtype=hidden1[0].dtype)
#             for raw_x in raw_xs:
#                 x = torch.tensor(raw_x, dtype=torch.float32).view(1,1,-1)
#                 out1, hidden1 = lstm1(x, hidden1)
#                 out2, hidden2 = lstm2(out1, hidden2)

#             if (out2 > 0.5 and raw_y[0] < 1) or (out2 < 0.5 and raw_y[0] > 0):
#                 err_num = err_num + 1
#                 loss = mse(out2, y)
#                 loss.backward()
#                 optimizer.step()
#                 # print(loss)
#                 # print('y_hat', out2)
#                 # print('raw_y', raw_y)
#         if err_num == 0:
#             print('reach full acc at epch', e, '\n')
#             break
#         # print(err_num)
#     if err_num != 0:
#         print('fail to reach full acc', '\n')


# %% [markdown]
# ## Word completion test

# %%
from custom_datasets.word_completion_dataset import Words_Completion_small as Dataset
global backward_count
backward_count = 0
from memory_profiler import profile

# class FF(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, x):
#         return x

#     @staticmethod
#     def backward(ctx, grad_out):
#         global backward_count
#         backward_count = backward_count + 1
#         print('back', backward_count)
#         return grad_out

@profile
def run():
    for t in range(1):
        lstm1 = torch.nn.LSTM(26,72)

        lstm2 = torch.nn.LSTM(72,72)
        linear = torch.nn.Linear(72, 26)
        model = torch.nn.ModuleList([lstm1, lstm2, linear])
        dataset = Dataset()
        optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)
        m = torch.nn.Softmax(dim=-1)
        mse = torch.nn.MSELoss()
        minError = 10
        totalTime = 0
        startTime = time.time()
        for e in range(1):
            dataset.shuffle()
            err_num = 0
            for xs, y in dataset:
                # print(len(xs))
                y = torch.tensor(y, dtype=torch.float32)
                hidden1 = (torch.zeros(1, 1, 72, dtype=torch.float32),
                        torch.zeros(1, 1, 72, dtype=torch.float32))
                hidden2 = (torch.zeros(1, 1, 72, dtype=torch.float32),
                        torch.zeros(1, 1, 72, dtype=torch.float32))
                
                for x in xs:
                    x = torch.tensor(x, dtype=torch.float32).view(1,1,-1)
                    for i in range(600):
                        # hidden1 = FF.apply(hidden1)
                        out1, hidden1 = lstm1(x, hidden1)
                        # print(hidden1[1].requires_grad)
                        # out1 = FF.apply(out1)
                        # hidden1 = FF.apply(hidden1)
                        out2, hidden2 = lstm2(out1, hidden2)
                        out3 = linear(out2)
                        
                outputIdx = torch.argmax(out3)
                if y[outputIdx] != 1:
                    # startTime = time.time()
                    err_num += 1
                    out = m(out3)
                    loss = mse(y, out)
                    loss.backward()
                    # for p in ps:
                    #     print(p.shape, p.grad)
                    optimizer.step()
                    # endTime = time.time()
                    # totalTime = totalTime + (endTime - startTime)
            endTime = time.time()
            print(endTime - startTime)
            if err_num == 0:
                print('achieved full accuracy at epoch', e, '\n')
                break
        if err_num < minError :
            minError = err_num
        if err_num != 0:
            print('fail achieve full accuracy', 'Highest Accuracy', (1 - minError / 10), '\n')
run()

# %%
# for t in range(10):
#     num_layers = 5
#     lstm = torch.nn.LSTM(26,26, num_layers=num_layers)
#     # lstm2 = torch.nn.LSTM(72,26)
#     # model = torch.nn.ModuleList([lstm1, lstm2])
#     dataset = Dataset()
#     optimizer = torch.optim.Adam(lstm.parameters(), lr = 0.0001)
#     m = torch.nn.Softmax()
#     mse = torch.nn.MSELoss()

#     for e in range(3000):
#         dataset.shuffle()
#         err_num = 0
#         for xs, y in dataset:
            
#             y = torch.tensor(y, dtype=torch.float32)
#             hidden = (torch.zeros(num_layers, 1, 26, dtype=torch.float32),
#                     torch.zeros(num_layers, 1, 26, dtype=torch.float32))
#             # hidden2 = (torch.zeros(1, 1, 26, dtype=torch.float32),
#             #         torch.zeros(1, 1, 26, dtype=torch.float32))
            
#             for x in xs:
#                 x = torch.tensor(x, dtype=torch.float32).view(1,1,-1)
#                 out, hidden = lstm(x, hidden)
#                 # out2, hidden2 = lstm2(out1, hidden2)
#             # print( out.shape)
#             # stop
#             outputIdx = torch.argmax(out)
#             if y[outputIdx] != 1:
#                 err_num += 1
#                 out = m(out)
#                 loss = mse(out, y)
#                 loss.backward()
#                 optimizer.step()
            
#             # out.backward(ex_g, retain_graph= True)
            
#             # else:
#             #     print(out[0][0][1], raw_y)
#         if err_num == 0:
#             print('finished training', e)
#             break
#     if err_num != 0:
#         print('fail to converge')

# %%
# for t in range(10):
#     num_layers = 5
#     lstm = torch.nn.LSTM(26,26, num_layers=num_layers)
#     # lstm2 = torch.nn.LSTM(72,26)
#     # model = torch.nn.ModuleList([lstm1, lstm2])
#     dataset = Dataset()
#     optimizer = torch.optim.Adam(lstm.parameters(), lr = 0.0001)
#     m = torch.nn.Softmax()
#     mse = torch.nn.MSELoss()

#     for e in range(3000):
#         dataset.shuffle()
#         err_num = 0
#         for xs, y in dataset:
            
#             y = torch.tensor(y, dtype=torch.float32)
#             hidden = (torch.zeros(num_layers, 1, 26, dtype=torch.float32),
#                     torch.zeros(num_layers, 1, 26, dtype=torch.float32))
#             # hidden2 = (torch.zeros(1, 1, 26, dtype=torch.float32),
#             #         torch.zeros(1, 1, 26, dtype=torch.float32))
            
#             for x in xs:
#                 x = torch.tensor(x, dtype=torch.float32).view(1,1,-1)
#                 out, hidden = lstm(x, hidden)
#                 # out2, hidden2 = lstm2(out1, hidden2)
#             # print( out.shape)
#             # stop
#             outputIdx = torch.argmax(out)
#             if y[outputIdx] != 1:
#                 err_num += 1
#                 out = m(out)
#                 loss = mse(out, y)
#                 loss.backward()
#                 optimizer.step()
            
#             # out.backward(ex_g, retain_graph= True)
            
#             # else:
#             #     print(out[0][0][1], raw_y)
#         if err_num == 0:
#             print('finished training', e)
#             break
#     if err_num != 0:
#         print('fail to converge')

# %%
# for t in range(10):
#     num_layers = 5
#     lstm = torch.nn.LSTM(26,26, num_layers=num_layers)
#     # lstm2 = torch.nn.LSTM(72,26)
#     # model = torch.nn.ModuleList([lstm1, lstm2])
#     dataset = Dataset()
#     optimizer = torch.optim.Adam(lstm.parameters(), lr = 0.0001)
#     m = torch.nn.Softmax()
#     mse = torch.nn.MSELoss()

#     for e in range(3000):
#         dataset.shuffle()
#         err_num = 0
#         for xs, y in dataset:
            
#             y = torch.tensor(y, dtype=torch.float32)
#             hidden = (torch.zeros(num_layers, 1, 26, dtype=torch.float32),
#                     torch.zeros(num_layers, 1, 26, dtype=torch.float32))
#             # hidden2 = (torch.zeros(1, 1, 26, dtype=torch.float32),
#             #         torch.zeros(1, 1, 26, dtype=torch.float32))
            
#             for x in xs:
#                 x = torch.tensor(x, dtype=torch.float32).view(1,1,-1)
#                 out, hidden = lstm(x, hidden)
#                 # out2, hidden2 = lstm2(out1, hidden2)
#             # print( out.shape)
#             # stop
#             outputIdx = torch.argmax(out)
#             if y[outputIdx] != 1:
#                 err_num += 1
#                 out = m(out)
#                 loss = mse(out, y)
#                 loss.backward()
#                 optimizer.step()
            
#             # out.backward(ex_g, retain_graph= True)
            
#             # else:
#             #     print(out[0][0][1], raw_y)
#         if err_num == 0:
#             print('finished training', e)
#             break
#     if err_num != 0:
#         print('fail to converge')


