import torch
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
from copy import deepcopy

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
        output_str = f'Best epoch : {bestEpoch} , ACC : {round(aucMax, 4)}'
        print(output_str)
        return reses['Acc'],output_str,aucMax

    # Function to prepare the model and optimizer
    def prepareModel(self):
        self.model = Model().cuda()
        self.opt = t.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0)

    # 排序筛选出困难负样本 4096 100
    def sort_hard_embedding(self, itmEmbeds, negEmbeds):
        """
        计算负样本与所有正样本(基因)的相似度，筛选困难负样本
        itmEmbeds: [num_genes, 128] - 所有基因嵌入
        negEmbeds: [batch_size, num_neg, 128] - 负样本嵌入
        """
        batch_size, num_neg, embed_dim = negEmbeds.shape
        num_genes = itmEmbeds.shape[0]
        
        # 重塑negEmbeds为 [batch_size * num_neg, 128] 以便批量计算
        negEmbeds_flat = negEmbeds.view(-1, embed_dim)  # [4096*100, 128]
        
        # 计算所有负样本与所有基因的相似度
        # 选项1: 点积相似度（当前使用）
        similarity_scores = torch.mm(negEmbeds_flat, itmEmbeds.T)

        # 应用指数函数增强相似度差异
        # similarity_scores = torch.exp(similarity_scores)
        
        # 重塑回原来的形状: [batch_size, num_neg, num_genes]  4096 100 1664  num_genes==1664
        # 每一个批次中，每一个负样本，与每一个正样本基因数据的相似度
        similarity_scores = similarity_scores.view(batch_size, num_neg, num_genes)
        
        # 对每个负样本，计算其与所有基因的平均相似度（整体困难程度）
        avg_similarities = torch.mean(similarity_scores, dim=2)  # [batch_size, num_neg]
        
        # 对每个样本的负样本按困难程度排序（相似度越高越困难）4096 100
        sorted_scores, sorted_indices = torch.sort(avg_similarities, dim=1, descending=True)

        return sorted_indices, sorted_scores



    # Function to train a single epoch
    # 返回损失值 更新参数
    def trainEpoch(self):
        self.model.train()
        trnLoader = self.handler.trnLoader
        print("开始生成负样本")
        # 这一步如果在DataHandler中就执行会牺牲随机性
        trnLoader.dataset.negSampling() #这一步非常消耗时间

        hard_loss_sum=0
        epLoss, epPreLoss = 0, 0
        bprLoss, bpr_loss,reg_loss ,regLoss,im_loss= 0, 0 , 0,0,0
        steps = trnLoader.dataset.__len__() // args.batch  # 步数=长度/批次数
        for i, tem in enumerate(trnLoader):
            data = deepcopy(self.handler.torchBiAdj).cuda()
            drugs, genes, labels , negs = tem
            drugs = drugs.long().cuda()
            genes = genes.long().cuda()
            labels = labels.long().cuda()
            negs = negs.long().cuda()
            usrEmbeds, itmEmbeds = self.model.forward_gcn(data)
            # 获取正样本和负样本的嵌入

            drugEmbeds = usrEmbeds[drugs]  # 药物嵌入 4096 128
            posEmbeds = itmEmbeds[genes]  # 正样本嵌入 4096 128
            negEmbeds = itmEmbeds[negs]  # 负样本嵌入 4096 100 128

            if (args.num_hard_neg > 0):
                # HaSa损失 - 筛选困难负样本
                hard_neg_indices, hard_neg_scores = self.sort_hard_embedding(itmEmbeds, negEmbeds)

                # 筛选出前num_hard_neg个最困难的负样本
                # 使用预定义的超参数
                num_hard_neg = min(args.num_hard_neg, args.num_neg)  # 不超过总负样本数
                print(f"Selecting top {num_hard_neg} hard negatives from {args.num_neg} negatives")

                # 获取困难负样本的索引 [batch_size, num_hard_neg]
                hard_indices_selected = hard_neg_indices[:, :num_hard_neg]

                # 获取对应的困难负样本分数 [batch_size, num_hard_neg]
                hard_neg_scores_selected = hard_neg_scores[:, :num_hard_neg]

                # 对困难负样本分数进行归一化和softmax处理
                # 方法1: 先L2归一化再softmax
                hard_scores_normalized = F.normalize(hard_neg_scores_selected, p=2, dim=1)  # L2归一化
                hard_prob = F.softmax(hard_scores_normalized, dim=1)  # softmax得到概率分布

                # 方法2: 直接对原始分数进行softmax（可选，取消注释使用）
                # hard_prob = F.softmax(hard_neg_scores_selected, dim=1)

                print(f"Hard neg scores shape: {hard_neg_scores_selected.shape}")
                print(f"Hard prob shape: {hard_prob.shape}")
                print(f"Hard prob sum check: {hard_prob.sum(dim=1)[:5]}")  # 检查概率和是否为1


                # 使用高级索引选择困难负样本的嵌入
                batch_indices = torch.arange(negEmbeds.size(0)).unsqueeze(1).expand(-1, num_hard_neg).cuda()
                # 得到困难负样本嵌入
                hard_negEmbeds = negEmbeds[batch_indices, hard_indices_selected]  # [batch_size, num_hard_neg, 128]

                hard_loss = self.model.batch_bias_hard(drugEmbeds, posEmbeds, hard_negEmbeds, hard_prob)
                regLoss = calcRegLoss(self.model) * args.reg
                hard_loss_sum+=float(hard_loss.item())

                loss = hard_loss+regLoss

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()


                # print(f"Original negEmbeds shape: {negEmbeds.shape}")
                # print(f"Selected hard negEmbeds shape: {hard_negEmbeds.shape}")
                # print(f"Hard indices shape: {hard_indices_selected.shape}")

            # 全局负采样损失 - 暂时先不管
            # if (args.num_neg != 0):
            #     # 批量计算分数差异
            #     # 通过 unsqueeze(1) 将正样本嵌入扩展维度，使其与负样本对齐
            #     posScores = innerProduct(drugEmbeds.unsqueeze(1), posEmbeds.unsqueeze(1))  # [batch_size, 1]
            #     # print(f"drugEmbeds.unsqueeze(1) shape: {drugEmbeds.unsqueeze(1).shape}")
            #     # print(f"hard_negEmbeds shape: {hard_negEmbeds.shape}")
            #
            #     # 使用困难负样本计算负样本分数（原始方法）
            #     # negScores = innerProduct(drugEmbeds.unsqueeze(1), hard_negEmbeds)  # [batch_size, num_hard_neg]
            #
            #     # 使用新的加权负样本分数计算函数
            #     weighted_negScores, aggregated_negScore = self.compute_weighted_negative_scores(
            #         drugEmbeds, hard_negEmbeds, hard_prob)
            #
            #     # 选择使用哪种负样本分数
            #     # 选项1: 使用加权的负样本分数
            #     negScores = weighted_negScores  # [batch_size, num_hard_neg]
            #
            #     # 选项2: 使用聚合的负样本分数（取消注释使用）
            #     # negScores = aggregated_negScore  # [batch_size, 1]
            #
            #     # 计算分数差异
            #     scoreDiff = posScores - negScores  # [batch_size, num_hard_neg]
            #
            #     # 计算 BPR 损失
            #     global_sample_loss = - (scoreDiff).sigmoid().log().sum() / args.batch
            #     # print(f"global_sample_loss shape: {global_sample_loss.shape}")
            #     # print(f"scoreDiff shape: {scoreDiff.shape}")
            #
            #     bprLoss += global_sample_loss
            #     regLoss = calcRegLoss(self.model) * args.reg
            #     loss = bprLoss + regLoss
            #     self.opt.zero_grad()
            #     loss.backward(retain_graph=True)
            #     self.opt.step()

            # 计算交叉熵损失
            ceLoss = self.model.calcLosses(drugs, genes, labels, self.handler.torchBiAdj, args.keepRate)
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

            bprLoss = 0
            regLoss = 0
        ret = dict()
        ret['Loss'] = epLoss / steps
        ret['preLoss'] = epPreLoss / steps
        ret['hard_loss'] = hard_loss_sum / steps
        ret['regLoss'] = regLoss / steps
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
    outputstr_list = list()

    epoch_list = []
    acc_list = []

    iteration_list = []
    end_acc_list = []
    output_str = None
    it_max=0
    aucMax=0
    for i in range(args.iteration):
        print('{}-th iteration'.format(i + 1))
        seed = args.seed + i
        config['seed'] = seed
        config['iteration'] = i + 1
        set_seed(seed)
        if args.data == 'LINCS':
            result = coach.external_test_run()
        else:
            result,output_str,aucMax = coach.run(i)  # 返回最终测试得到的reses['Acc']
        iteration_list.append(i)
        end_acc_list.append(result)
        results.append(result)
        aucMax_list.append(aucMax)
        outputstr_list.append(output_str)
        if aucMax > it_max:
            it_max = aucMax


    plt.plot(iteration_list, end_acc_list)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    # plt.savefig('/home/huangpeng/DGCL-main/dgcl.png')

    avg_r = np.mean(np.array(results), axis=0)
    avg_aucMax = np.mean(np.array(aucMax_list), axis=0)
    std_r = np.std(results, axis=0)  # 求标准差
    print('test results: ')
    print(results)
    print('best epoch results: ')
    print(outputstr_list)
    print('平均值: {}'.format(avg_r))
    print('平均最大值: {}'.format(avg_aucMax))
    print('iteration最大值: {}'.format(it_max))

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
