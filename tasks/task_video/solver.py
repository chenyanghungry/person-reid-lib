from lib.network.solver_factory import TaskSolverBase
from architecture import NetClient, ModelClient
__all__ = ['Solver']

"""
Solver一般都是保存网络，这里继承了基础网络TaskSolverBase，这里的两个函数重合了，注意是执行那个
"""
class Solver(TaskSolverBase):
    def const_options(self):
        # ------train-------
        self.display_step = 1
        self.instance_num = 10 if self.manager.device['name'] != 'pc' else 3
        self.sample_num = 2
        self.depth = 8 if self.manager.device['name'] != 'pc' else 2

        self.train_dataloder_type = 'All'
        self.minframes = {'train': self.depth, 'test': 1}

        # ------test--------
        self.test_batch_size = self.manager.device['test_batch_size']

    def init_options(self):
        # ------option------
        self.use_flow = False
        self.save_model = True
        self.reuse_model = False
        self.store_search_result = False
        self.net_client = NetClient
        self.model_client = ModelClient  #这两行代码需要注意，Python中类初始化是需要（）的，这里没有，类似将该类赋值给对应的变量，类似起别名
