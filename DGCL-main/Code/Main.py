from matplotlib import pyplot as plt
import matplotlib
import Utils.TimeLogger as logger
from Utils.TimeLogger import log
from Params import args
from Model_sparse import Model
from DataHandler import DataHandler
from Utils.Utils import *
import os
import torch.nn.functional as F

from sklearn.metrics import accuracy_score

import numpy as np
import random


# Function to set random seed for reproducibility
#通过设置一个固定的种子值，我们可以确保每次实验的初始化和随机过程相同，从而使得实验的结果可重复
def set_seed(seed):
    print(seed)
    # 设置Python随机种子
    random.seed(seed)
    # 设置NumPy随机种子
    np.random.seed(seed)
    # 设置PyTorch随机种子
    t.manual_seed(seed)

    if t.cuda.is_available():
        t.cuda.manual_seed(seed)
        #避免 cuDNN 根据硬件自动选择高效但不确定的算法，保证每次运行的算法一致
        #降低训练速度
        t.backends.cudnn.benchmark = False  #是否使用 cuDNN 的自动算法优化功能
        #消除算法本身的不确定性（如某些浮点运算的舍入误差），确保 GPU 计算结果严格一致
        #可能牺牲部分性能
        t.backends.cudnn.deterministic = True  #是否强制 cuDNN 使用确定性算法


# Define the Coach class for model training and evaluation
class Coach:
    def __init__(self, handler):
        self.handler = handler

        print('DRUG    ', args.drug, 'GENE', args.gene)
        #数据集长度等价于DGI的长度
        print('NUM  OF   INTERACTIONSSS', self.handler.trnLoader.dataset.__len__())
        self.metrics = dict()  #哈希表
        mets = ['Loss', 'preLoss', 'Acc']
        for met in mets:
            self.metrics['Train' + met] = list()
            self.metrics['Test' + met] = list()

    # Function to create a formatted print statement
    def makePrint(self, name, ep, reses, save):
        ret = 'Epoch %d/%d, %s: ' % (ep, args.epoch, name)
        for metric in reses:
            val = reses[metric]
            ret += '%s = %.4f, ' % (metric, val)
            tem = name + metric
            if save and tem in self.metrics:
                self.metrics[tem].append(val)
        ret = ret[:-2] + '  '
        return ret

    # Function to perform external testing
    def external_test_run(self):
        self.prepareModel()
        log('Model Prepared')
        if args.load_model != None:
            self.loadModel()
        reses = self.testEpoch()
        log(self.makePrint('Test', args.epoch, reses, True))
        return reses['Acc']

    # Function to train and evaluate the model
    def run(self,i):
        self.prepareModel()
        log('Model Prepared')
        if args.load_model != None:  # 有模型需要加载
            self.loadModel()
            # 已经训练批次*每轮批次的epoch
            stloc = len(self.metrics['TrainLoss']) * args.tstEpoch - (args.tstEpoch - 1)
        else:
            stloc = 0
            log('Model Initialized')
        epoch_list = []
        acc_list = []

        aucMax = 0
        bestEpoch = 0

        for ep in range(stloc, args.epoch):
            tstFlag = (ep % args.tstEpoch == 0)
            reses = self.trainEpoch()
            train_loss = reses
            # 记录训练结果
            log(self.makePrint('Train', ep, reses, tstFlag))
            if tstFlag:
                reses = self.testEpoch()
                test_r = reses

                if reses['Acc'] > aucMax:
                    aucMax = reses['Acc']
                    bestEpoch = ep

                # 记录测试结果
                log(self.makePrint('Test', ep, reses, tstFlag))
            logs = {'loss_all': train_loss['Loss'], 'loss_pre': train_loss['preLoss'],
                    'test_acc': test_r['Acc']}
            epoch_list.append(ep)
            acc_list.append(test_r['Acc'])
        # wandb.log(logs)
        plt.plot(epoch_list, acc_list)
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        #这里没有权限来保存图片
#        plt.savefig('/home/huangpeng/DGCL-main/dgcl{}.png'.format(i))

        reses = self.testEpoch()
        log(self.makePrint('Test', args.epoch, reses, True))
        self.save_model('{}'.format(config['iteration']))
        # 每轮itration结束补充输出一个最好的结果
        print('Best epoch : ', bestEpoch, ' , AUC : ', aucMax)
        return reses['Acc'],aucMax

    # Function to prepare the model and optimizer
    def prepareModel(self):
        self.model = Model().cuda()
        self.opt = t.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0)

    # Function to train a single epoch
    # 返回损失值 更新参数
    def trainEpoch(self):
        self.model.train()
        trnLoader = self.handler.trnLoader
        epLoss, epPreLoss = 0, 0
        steps = trnLoader.dataset.__len__() // args.batch  # 步数=长度/批次数
        for i, tem in enumerate(trnLoader):
            drugs, genes, labels = tem
            drugs = drugs.long().cuda()
            genes = genes.long().cuda()
            labels = labels.long().cuda()
            ceLoss= self.model.calcLosses(drugs, genes, labels, self.handler.torchBiAdj, args.keepRate)
            # sslLoss = sslLoss * args.ssl_reg
            regLoss = calcRegLoss(self.model) * args.reg
            # loss = ceLoss + regLoss + sslLoss
            loss = ceLoss + regLoss
            epLoss += loss.item()
            epPreLoss += ceLoss.item()
            # 清空优化器中的所有梯度，使得下一次反向传播时不会受到之前梯度的影响
            self.opt.zero_grad()
            # 反向传播 沿着计算图计算出梯度
            loss.backward()
            # 使用计算得到的梯度和学习率
            # 使用优化算法（例如 SGD、Adam、RMSprop 等）来更新模型参数
            self.opt.step()
        ret = dict()
        ret['Loss'] = epLoss / steps
        ret['preLoss'] = epPreLoss / steps
        return ret

    # Function to test a single epoch
    # 计算出ACC
    def testEpoch(self):
        self.model.eval()
        tstLoader = self.handler.tstLoader
        i = 0
        for tem in tstLoader:
            i += 1
            drugs, genes, labels = tem
            # 将从数据集获得的张量转移到Gpu上面，默认第一个Gpu
            drugs = drugs.long().cuda()
            genes = genes.long().cuda()
            labels = labels.long().cuda()
            # predict(self, adj, drugs, genes)
            pre = self.model.predict(self.handler.torchBiAdj, drugs, genes)
            # dim=1 指定了 softmax 操作沿着第 1 维（即类别维度）进行
            pre = F.log_softmax(pre, dim=1)
            # 选出可能性最大的类别 keepdim=True 保证返回的张量保持相同的维度
            pre = pre.data.max(1, keepdim=True)[1].detach().cpu()
            # labels脱离计算图，不参与反向传播，并且张量从 GPU 转移到 CPU
            labels = labels.detach().cpu()
            epAcc = accuracy_score(labels, pre)
        ret = dict()
        ret['Acc'] = epAcc
        return ret

    # Function to load a pre-trained model
    def loadModel(self):
        self.model.load_state_dict(t.load('../Models/' + args.load_model + '.pkl'))
        self.opt = t.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0)
        log('Model Loaded')

    # Function to save the trained model
    def save_model(self, model_path):
        # 使用 wandb.run.dir 获取当前 WandB 实验的根目录
        model_parent_path = "D:\\桌面\\研\\论文\\实验代码\\DGCL-main\\DGCL-main\\Models"
        # model_parent_path = os.path.join(wandb.run.dir, 'ckl')
        # if not os.path.exists(model_parent_path):
        #     os.mkdir(model_parent_path)
        # # 将模型参数保存到指定的模型路径上（网络层的权重、偏置）
        # t.save(self.model.state_dict(), '{}/{}_model.pkl'.format(model_parent_path, model_path))


# Main execution block
if __name__ == '__main__':

    matplotlib.use('Agg')
    try:
        import wandb
    except ModuleNotFoundError:
        print("wandb is not installed, skipping related functionality.")


    if args.is_debug is True:
        print("DEBUGGING MODE - Start without wandb")
        # debug模式下不需要wandb
        #wandb.init(mode="disabled")
    # else:
    # 记录配置与项目名
    # wandb.init(project='HC', config=args)
    # '.' 表示将当前目录 将当前目录中的所有代码上传
    # wandb.run.log_code(".")


    use_cuda = args.gpu >= 0 and t.cuda.is_available()
    device = 'cuda:{}'.format(args.gpu) if use_cuda else 'cpu'
    if use_cuda:
        t.cuda.set_device(device)
    args.device = device

    logger.saveDefault = True

    log('Start')
    handler = DataHandler()
    handler.LoadData()
    log('Load Data')

    coach = Coach(handler)
    config = dict()
    results = list()
    aucMax_list = list()

    epoch_list = []
    acc_list = []

    iteration_list = []
    end_acc_list = []
    aucMax=0;
    for i in range(args.iteration):
        print('{}-th iteration'.format(i + 1))
        seed = args.seed + i
        config['seed'] = seed
        config['iteration'] = i + 1
        set_seed(seed)
        if args.data == 'LINCS':
            result = coach.external_test_run()
        else:
            result,aucMax = coach.run(i)  # 返回最终测试得到的reses['Acc']
        iteration_list.append(i)
        end_acc_list.append(result)
        results.append(result)
        aucMax_list.append(aucMax)

    plt.plot(iteration_list, end_acc_list)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    # plt.savefig('/home/huangpeng/DGCL-main/dgcl.png')

    avg_r = np.mean(np.array(results), axis=0)
    avg_aucMax = np.mean(np.array(aucMax_list), axis=0)
    std_r = np.std(results, axis=0)  # 求标准差
    print('test results: ')
    print(results)
    print(aucMax_list)
    print('平均值: {}'.format(avg_r))
    print('平均最大值: {}'.format(avg_aucMax))
    print('方差: {}'.format(std_r))

    results.append(avg_r)
    results.append(std_r)

    # results_parent_path = "D:\\桌面\\研\\论文\\实验代码\\DGCL-main\\DGCL-main\\results"
    # # results_parent_path = os.path.join(wandb.run.dir, 'results')
    # if not os.path.exists(results_parent_path):
    #     os.mkdir(results_parent_path)
    # np.savetxt('{}/{}_result.txt'.format(results_parent_path, args.data), np.array(results), delimiter=",", fmt='%f')

    print('result saved!!!')
    #wandb.finish()
