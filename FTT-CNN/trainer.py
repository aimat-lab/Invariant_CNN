import torch
import numpy as np
import matplotlib.pyplot as plt

from typing import List, Dict, Optional
from tqdm import tqdm

from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader

from utils import manifold_shift_image, flip_image



class Trainer:
    def __init__(
        self, 
        model,
        config : Dict,
    ):
        '''
        config:
            name 
            device
            loss
            batch
            epochs
            lr
            scheduler_step
        '''
        super().__init__()
        self.config = config
        self.model = model.to(self.config['device'])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['lr'])#, weight_decay= 1.e-04)
        self.scheduler = StepLR(self.optimizer, step_size=config['scheduler_step'], gamma=0.7)
		
    def _save(self, extend_name=''):
        try:
            state_dict = self.model.module.state_dict()
        except AttributeError:
            state_dict = self.model.state_dict()    
        torch.save(state_dict, './models/'+self.config['name']+extend_name+'.pt')
        print('Model saved')	
	
    @staticmethod	
    def _plot(
        vals: Dict, 
        x_axis: Optional[List] = None, 
        name: str = 'plot', 
        save: bool = True,
    ):
        n_plots = len(vals.keys())
        fig, axs = plt.subplots(nrows=1, ncols=n_plots, sharex=True)
        if n_plots==1:
            key = list(vals.keys())[0]
            if x_axis is None or len(x_axis)!=len(vals[key]):
                xs = np.arange(len(vals[key]))
            else:
                xs = x_axis
            axs.plot(xs, vals[key], label=key)
            axs.legend()
        else:
            for i, key in enumerate(vals.keys()):
                if x_axis is None or len(x_axis)!=len(vals[key]):
                    xs = np.arange(len(vals[key]))
                else:
                    xs = x_axis
                axs[i].plot(xs, vals[key], label=key)
                axs[i].legend()
        if save:
            fig.savefig(f'./outputs/{name}.png')
        else:
            plt.show()
        plt.close()
		
    def fit(
        self, 
        train_data: Dataset, 
        val_data: Optional[Dataset] = None, 
        plot: bool = True,
    ):
        train_loader = DataLoader(train_data, batch_size=self.config['batch'], shuffle=True)
        if val_data is not None:
            val_loader = DataLoader(val_data, batch_size=self.config['batch'], shuffle=False)

        if self.config.get('parallelize', False):
            self.model = torch.nn.DataParallel(self.model)
            print(f"Using {torch.cuda.device_count()} GPUs in parallel")
        
        losses = {'train' : [], 'val' : []}
        time_steps = []
        best_val = (np.inf, 0)
        for epoch in range(self.config['epochs']):
            self.model.train()
            print(f'-Epoch {epoch + 1}:')
            cum_loss = []
            for X, y in tqdm(train_loader):
                X = X.to(self.config['device'])
                y = y.to(self.config['device'])
                
                self.optimizer.zero_grad()
                pred = self.model(X).squeeze()
                loss = self.config['loss'](pred, y)
                loss.backward()
                self.optimizer.step()
                cum_loss.append(loss.item())
                
            if self.optimizer.param_groups[0]['lr'] > 1e-6:
                self.scheduler.step()
            if self.optimizer.param_groups[0]['lr'] < 1e-6:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = 1e-5
                    
            losses['train'].append(torch.tensor(cum_loss).mean())
            print(f"\t Train loss: {losses['train'][-1]}")
            
            if val_data is not None:
                self.test(val_loader, 'val', losses)
                if (epoch+1)>40:  ## avoid saving best model among those with strongly oscillating losses
                    if losses['val'][-1] < best_val[0]:
                        best_val = losses['val'][-1], epoch+1
                        self._save(f'_best')
            if (epoch+1)%10==0:
                self._save(f'_ep{epoch}')
                    
        print(f'best validation error : {best_val[0]} at epoch {best_val[1]}')
        self._save('_last')
        if plot:
            self._plot(losses, name = self.config['name'], x_axis=time_steps)	
	
    def test(
        self, 
        dataloader: DataLoader, 
        partition: str = 'test', 
        losses: Optional[Dict] = None,
        shifts: bool = False,
        flips: bool = False,
    ):
        self.model.eval()
        preds, targets = [], []
        flag = True
        if losses is None:
            flag = False
            losses = {partition : []}
        with torch.no_grad():
            cum_loss = []
            for X, y in tqdm(dataloader):
                X = X.to(self.config['device'])
                y = y.to(self.config['device'])
                if shifts:
                    X = manifold_shift_image(X)
                if flips:
                    X = flip_image(X)
                
                pred = self.model(X)
                loss = self.config['loss'](pred, y)
                cum_loss.append(loss.item())
                
                preds.append(pred)
                targets.append(y)
            losses[partition].append(torch.tensor(cum_loss).mean())
            if flag:
                print(f"\t {partition} loss: {losses[partition][-1]}")
            else:
                return losses[partition][-1].item(), torch.concat(preds), torch.concat(targets)
