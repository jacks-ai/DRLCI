import torch
import torch as t
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
# 通过设置一个固定的种子值，我们可以确保每次实验的初始化和随机过程相同，从而使得实验的结果可重复
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
        # 避免 cuDNN 根据硬件自动选择高效但不确定的算法，保证每次运行的算法一致
        # 降低训练速度
        t.backends.cudnn.benchmark = False  # 是否使用 cuDNN 的自动算法优化功能
        # 消除算法本身的不确定性（如某些浮点运算的舍入误差），确保 GPU 计算结果严格一致
        # 可能牺牲部分性能
        t.backends.cudnn.deterministic = True  # 是否强制 cuDNN 使用确定性算法


# Define the Coach class for model training and evaluation
class Coach:
    def __init__(self, handler):
        self.handler = handler

        print('DRUG    ', args.drug, 'GENE', args.gene)
        # 数据集长度等价于DGI的长度
        print('NUM  OF   INTERACTIONSSS', self.handler.trnLoader.dataset.__len__())
        self.metrics = dict()  # 哈希表
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
    def run(self, i):
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
        # 这里没有权限来保存图片
        #        plt.savefig('/home/huangpeng/DGCL-main/dgcl{}.png'.format(i))

        reses = self.testEpoch()
        log(self.makePrint('Test', args.epoch, reses, True))
        self.save_model('{}'.format(config['iteration']))
        # 每轮itration结束补充输出一个最好的结果
        output_str = f'Best epoch : {bestEpoch} , ACC : {round(aucMax, 4)}'
        print(output_str)
        return reses['Acc'], output_str, aucMax

    # Function to prepare the model and optimizer
    def prepareModel(self):
        self.model = Model().cuda()
        self.is_data_parallel = False

        # 如果启用多GPU并且有多个GPU可用，使用DataParallel
        if args.multi_gpu and t.cuda.device_count() > 1:
            gpu_list = [int(x) for x in args.gpu_list.split(',')]
            available_gpus = [gpu for gpu in gpu_list if gpu < t.cuda.device_count()]
            if len(available_gpus) > 1:
                print(f"Wrapping model with DataParallel on GPUs: {available_gpus}")
                self.model = t.nn.DataParallel(self.model, device_ids=available_gpus)
                self.is_data_parallel = True

        self.opt = t.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0)

    def get_model(self):
        """获取原始模型，处理DataParallel包装问题"""
        if self.is_data_parallel:
            return self.model.module
        else:
            return self.model

    # 排序筛选出困难负样本 4096 100
    def sort_hard_embedding(self, itmEmbeds, negEmbeds):
        """
        计算负样本与所有正样本(基因)的相似度，筛选困难负样本
        使用双GPU并行计算，避免显存爆炸
        itmEmbeds: [num_genes, 128] - 所有基因嵌入
        negEmbeds: [batch_size, num_neg, 128] - 负样本嵌入
        """
        batch_size, num_neg, embed_dim = negEmbeds.shape
        num_genes = itmEmbeds.shape[0]

        # 重塑negEmbeds为 [batch_size * num_neg, 128] 以便批量计算
        negEmbeds_flat = negEmbeds.view(-1, embed_dim)  # [4096*100, 128]
        total_neg_samples = negEmbeds_flat.shape[0]

        print(f"Computing similarity for {total_neg_samples} neg samples vs {num_genes} genes")
        print(f"Estimated memory needed: {(total_neg_samples * num_genes * 4) / 1024 ** 3:.2f} GB")

        if args.multi_gpu and torch.cuda.device_count() > 1:
            # 双GPU并行方案
            gpu_list = [int(x) for x in args.gpu_list.split(',')]
            available_gpus = [gpu for gpu in gpu_list if gpu < torch.cuda.device_count()][:2]  # 只使用前两个GPU
            print(f"Using dual GPUs : {available_gpus}")

            # 检查GPU显存状态
            for gpu_id in available_gpus:
                gpu_memory_total = torch.cuda.get_device_properties(gpu_id).total_memory / 1024 ** 3
                gpu_memory_allocated = torch.cuda.memory_allocated(gpu_id) / 1024 ** 3
                gpu_memory_free = gpu_memory_total - gpu_memory_allocated
                print(f"GPU {gpu_id}: {gpu_memory_free:.1f}GB free / {gpu_memory_total:.1f}GB total")

            # 将数据分成两半
            mid_point = total_neg_samples // 2
            neg_part1 = negEmbeds_flat[:mid_point]  # 前半部分
            neg_part2 = negEmbeds_flat[mid_point:]  # 后半部分

            # 将itmEmbeds复制到两个GPU
            itmEmbeds_gpu0 = itmEmbeds.to(f'cuda:{available_gpus[0]}')
            itmEmbeds_gpu1 = itmEmbeds.to(f'cuda:{available_gpus[1]}')

            # GPU 0 计算前半部分
            neg_part1_gpu0 = neg_part1.to(f'cuda:{available_gpus[0]}')
            print(f"GPU {available_gpus[0]} processing {neg_part1.shape[0]} samples...")
            with torch.cuda.device(available_gpus[0]):
                similarity_part1 = t.mm(neg_part1_gpu0, itmEmbeds_gpu0.T)
                # 清理GPU 0上的中间变量
                del neg_part1_gpu0, itmEmbeds_gpu0
                torch.cuda.empty_cache()
                # 将第一部分结果移到GPU 1，为合并做准备
                similarity_part1 = similarity_part1.to(f'cuda:{available_gpus[1]}')

            # GPU 1 计算后半部分
            neg_part2_gpu1 = neg_part2.to(f'cuda:{available_gpus[1]}')
            print(f"GPU {available_gpus[1]} processing {neg_part2.shape[0]} samples...")
            with torch.cuda.device(available_gpus[1]):
                similarity_part2 = t.mm(neg_part2_gpu1, itmEmbeds_gpu1.T)
                # 清理GPU 1上的中间变量，为合并腾出空间
                del neg_part2_gpu1, itmEmbeds_gpu1
                torch.cuda.empty_cache()

                # 在GPU 1上进行合并操作，避免主GPU显存爆炸
                print(f"GPU {available_gpus[1]} merging results....")
                similarity_scores = t.cat([similarity_part1, similarity_part2], dim=0)

                # 清理合并用的临时变量
                del similarity_part1, similarity_part2
                torch.cuda.empty_cache()

                # 合并完成后移回主GPU
                similarity_scores = similarity_scores.to(negEmbeds.device)

            # 清理GPU缓存
            torch.cuda.empty_cache()
            print("✅ Dual GPU computation completed!")

        else:
            # 单GPU方案（原始计算）  DGIdb使用多GPU empty_cache()会出现异常
            print("⚠️ Multi-GPU not enabled, using single GPU computation")
            similarity_scores = t.mm(negEmbeds_flat, itmEmbeds.T)

        # 重塑回原来的形状: [batch_size, num_neg, num_genes]
        similarity_scores = similarity_scores.view(batch_size, num_neg, num_genes)

        # 对每个负样本，计算其与所有基因的平均相似度（整体困难程度）
        avg_similarities = t.mean(similarity_scores, dim=2)  # [batch_size, num_neg]

        # 使用数值稳定的exp计算
        avg_similarities_clamped = t.clamp(avg_similarities, min=args.score_clamp_min, max=args.score_clamp_max)
        avg_exp = t.exp(avg_similarities_clamped)

        # 对每个样本的负样本按困难程度排序（相似度越高越困难）
        sorted_scores, sorted_indices = t.sort(avg_exp, dim=1, descending=True)

        return sorted_indices, sorted_scores

    # Function to train a single epoch
    # 返回损失值 更新参数
    def trainEpoch(self):
        self.model.train()
        trnLoader = self.handler.trnLoader

        hard_loss_sum = 0
        epLoss, epPreLoss = 0, 0
        bprLoss, bpr_loss, reg_loss, regLoss, im_loss = 0, 0, 0, 0, 0
        # 数据集长度
        len__ = trnLoader.dataset.__len__()
        steps = len__ // args.batch  # 步数=长度/批次数
        for i, tem in enumerate(trnLoader):
            data = deepcopy(self.handler.torchBiAdj).cuda()
            drugs, genes, labels, negs = tem
            drugs = drugs.long().cuda()
            genes = genes.long().cuda()
            labels = labels.long().cuda()
            negs = negs.long().cuda()
            usrEmbeds, itmEmbeds = self.get_model().forward_gcn(data)
            # 获取正样本和负样本的嵌入
            drugEmbeds = usrEmbeds[drugs]  # 药物嵌入 4096 128
            posEmbeds = itmEmbeds[genes]  # 正样本嵌入 4096 128
            negEmbeds = itmEmbeds[negs]  # 负样本嵌入 4096 100 128

            # 全局负采样
            if(args.num_neg != 0):
                # BPR_0.706 + 0.03
                usrEmbeds, itmEmbeds = self.model.forward_gcn(data)
                # 获取正样本和负样本的嵌入
                ancEmbeds = usrEmbeds[drugs]  # 用户嵌入
                posEmbeds = itmEmbeds[genes]  # 正样本嵌入
                negEmbeds = itmEmbeds[negs]  # 所有负样本嵌入

                # 批量计算分数差异
                # ancEmbeds: [batch_size, embed_dim]
                # posEmbeds: [batch_size, embed_dim]
                # negEmbeds: [batch_size, num_neg, embed_dim]
                # 通过 unsqueeze(1) 将正样本嵌入扩展维度，使其与负样本对齐
                posScores = innerProduct(ancEmbeds.unsqueeze(1), posEmbeds.unsqueeze(1))  # [batch_size, 1]
                negScores = innerProduct(ancEmbeds.unsqueeze(1), negEmbeds)  # [batch_size, num_neg]
                # 计算分数差异
                scoreDiff = posScores - negScores  # [batch_size, num_neg]
                # 计算 BPR 损失
                bprLoss += - (scoreDiff).sigmoid().log().sum() / args.batch
                # 正则化损失
                regLoss += calcRegLoss(self.model) * args.reg
                loss = bprLoss + regLoss
                bpr_loss += float(bprLoss)
                reg_loss += float(regLoss)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

            # 计算交叉熵损失
            ceLoss = self.get_model().calcLosses(drugs, genes, labels, self.handler.torchBiAdj, args.keepRate)
            # sslLoss = sslLoss * args.ssl_reg
            regLoss = calcRegLoss(self.model) * args.reg
            # loss = ceLoss + regLoss + sslLoss
            loss = ceLoss + regLoss
            # 优化GPU->CPU传输
            epLoss += loss.detach().item()
            epPreLoss += ceLoss.detach().item()
            # 清空优化器中的所有梯度，使得下一次反向传播时不会受到之前梯度的影响
            self.opt.zero_grad()
            # 反向传播 沿着计算图计算出梯度
            loss.backward()
            # 梯度裁剪，防止梯度爆炸
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=args.clip_grad_norm)
            # 使用计算得到的梯度和学习率
            # 使用优化算法（例如 SGD、Adam、RMSprop 等）来更新模型参数
            self.opt.step()
            # bprLoss = 0
            regLoss = 0
        ret = dict()
        ret['Loss'] = epLoss / steps
        ret['preLoss'] = epPreLoss / steps
        ret['hard_loss_sum'] = hard_loss_sum / steps
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
            pre = self.get_model().predict(self.handler.torchBiAdj, drugs, genes)
            # dim=1 指定了 softmax 操作沿着第 1 维（即类别维度）进行
            pre = F.log_softmax(pre, dim=1)
            # 选出可能性最大的类别，优化GPU->CPU传输
            pre = pre.data.max(1, keepdim=True)[1]
            # 批量转移到CPU，减少传输次数
            pre = pre.detach().cpu()
            labels = labels.detach().cpu()
            epAcc = accuracy_score(labels, pre)
        ret = dict()
        ret['Acc'] = epAcc
        return ret

    # Function to load a pre-trained model
    def loadModel(self):
        # 处理DataParallel包装的模型加载
        if self.is_data_parallel:
            self.model.module.load_state_dict(t.load('../Models/' + args.load_model + '.pkl'))
        else:
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
        # # 处理DataParallel包装的模型保存
        # if self.is_data_parallel:
        #     t.save(self.model.module.state_dict(), '{}/{}_model.pkl'.format(model_parent_path, model_path))
        # else:
        #     t.save(self.model.state_dict(), '{}/{}_model.pkl'.format(model_parent_path, model_path))


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
        # wandb.init(mode="disabled")
    # else:
    # 记录配置与项目名
    # wandb.init(project='HC', config=args)
    # '.' 表示将当前目录 将当前目录中的所有代码上传
    # wandb.run.log_code(".")

    use_cuda = args.gpu >= 0 and t.cuda.is_available()

    if args.multi_gpu and t.cuda.device_count() > 1:
        gpu_list = [int(x) for x in args.gpu_list.split(',')]
        available_gpus = [gpu for gpu in gpu_list if gpu < t.cuda.device_count()]
        print(f"Multi-GPU mode enabled. Available GPUs: {available_gpus}")
        device = 'cuda:{}'.format(available_gpus[0])  # 主GPU
        args.device = device
        args.available_gpus = available_gpus
    else:
        device = 'cuda:{}'.format(args.gpu) if use_cuda else 'cpu'
        args.device = device

    if use_cuda:
        t.cuda.set_device(device)

    print(f"Primary device: {device}")
    print(f"Total GPU count: {t.cuda.device_count()}")

    # 显示GPU内存信息
    if use_cuda:
        for i in range(t.cuda.device_count()):
            gpu_memory = t.cuda.get_device_properties(i).total_memory / 1024 ** 3
            print(f"GPU {i}: {t.cuda.get_device_name(i)}, {gpu_memory:.1f} GB")

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
    it_max = 0
    aucMax = 0
    for i in range(args.iteration):
        print('{}-th iteration'.format(i + 1))
        seed = args.seed + i
        config['seed'] = seed
        config['iteration'] = i + 1
        set_seed(seed)
        if args.data == 'LINCS':
            result = coach.external_test_run()
        else:
            result, output_str, aucMax = coach.run(i)  # 返回最终测试得到的reses['Acc']
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
    # wandb.finish()
