import os
import logging
from collections import OrderedDict
from utils import get_db_codes_and_targets, save_tensor, read_tensor
from tqdm import tqdm
import numpy as np
import torch

def train(datahub, model, loss_fn, optimizer, lr_scheduler, config, 
          compute_err=True, evaluator=None, monitor=None, writer=None):
    model = model.to(config.device)
    if config.queue_begin_epoch != np.inf:
        logging.info("register queue")
        model.register_queue(enqueue_size=config.batch_size, device=config.device)
    for epoch in range(config.epoch_num):
        use_queue = (epoch >= config.queue_begin_epoch)
        epoch_loss, epoch_quant_err, batch_num = 0, 0, len(datahub.train_loader)
        for i, (train_data, _) in enumerate(tqdm(datahub.train_loader, desc="epoch %d" % epoch)):
            global_step = i + epoch * batch_num
            model.codebook_normalization()
            view1_data = train_data[0].to(config.device)
            view2_data = train_data[1].to(config.device)
            view3_data = train_data[2].to(config.device)
            view4_data = train_data[3].to(config.device)
            view5_data = train_data[4].to(config.device)
            view6_data = train_data[5].to(config.device)
            if lr_scheduler is not None:
                curr_lr = lr_scheduler.step()
                if writer is not None:
                    writer.add_scalar('lr', curr_lr, global_step)
            optimizer.zero_grad()

            view1_err, view2_err = -1, -1
            return_list = model(view1_data, compute_err=compute_err)
            if compute_err:
                x1, view1_feats, view1_soft_codes, view1_err = return_list
            else:
                x1, view1_feats, view1_soft_codes = return_list

            return_list = model(view2_data, compute_err=compute_err)
            if compute_err:
                x2, view2_feats, view2_soft_codes, view2_err = return_list
            else:
                x2, view2_feats, view2_soft_codes = return_list
            return_list = model(view3_data, only_feats=True,compute_err=compute_err)
            x3 = return_list
            return_list = model(view4_data, only_feats=True,compute_err=compute_err)
            x4 = return_list
            return_list = model(view5_data, only_feats=True,compute_err=compute_err)
            x5 = return_list
            return_list = model(view6_data, only_feats=True,compute_err=compute_err)
            x6 = return_list
            if compute_err:
                quant_err = (view1_err + view2_err) / 2
                epoch_quant_err += quant_err.item()
                if writer is not None:
                    writer.add_scalar('quant_err', quant_err.item(), global_step)

            queue_feats = model.get_queue_feats() if use_queue else None

            loss = loss_fn(x3,x4,x5,x6,view1_feats, view2_feats, queue_feats, 
                           view1_soft_codes, view2_soft_codes, model.codebooks, 
                           global_step=global_step, writer=writer)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

            if config.queue_begin_epoch != np.inf:
                model.dequeue_enqueue(view1_soft_codes.detach())
                model.dequeue_enqueue(view2_soft_codes.detach())

        logging.info("epoch %d: avg loss=%f, avg quantization error=%f" % 
                     (epoch, epoch_loss / batch_num, epoch_quant_err / batch_num))

        if evaluator is not None:
            if (epoch+1) % config.eval_interval == 0:
                logging.info("begin to evaluate model")
                evaluator.set_codebooks(codebooks=model.codebooks)
                db_codes, db_targets = get_db_codes_and_targets(datahub.database_loader, 
                                                                model, device=config.device)
                evaluator.set_db_codes(db_codes=db_codes)
                evaluator.set_db_targets(db_targets=db_targets)
                logging.info("compute mAP")
                val_mAP = evaluator.MAP(datahub.test_loader, model, topK=config.topK)
                logging.info("val mAP=%f" % val_mAP)
                if writer is not None:
                    writer.add_scalar("val_mAP", val_mAP, global_step)
                if monitor:
                    is_break, is_lose_patience = monitor.update(val_mAP)
                    if is_break:
                        logging.info("early stop")
                        break
                    if not is_lose_patience:
                        logging.info("save the best model, db_codes and db_targets")
                        model_parameters = OrderedDict()
                        for name, params in model.state_dict().items():
                            if name.find('queue') == -1: 
                                model_parameters[name] = params
                        
                        if (epoch+1) % 1000 == 0:
                            torch.save(model_parameters,
                                       os.path.join(config.checkpoint_root, 'model.cpt'))
                            save_tensor(db_codes,
                                        os.path.join(config.checkpoint_root, 'db_codes.npy'))
                            save_tensor(db_targets,
                                        os.path.join(config.checkpoint_root, 'db_targets.npy'))
                        logging.info("finish saving")
                        
    if config.queue_begin_epoch != np.inf:
        model.release_queue()
        logging.info("free the queue memory")
    logging.info("finish trainning at epoch %d" % epoch)


def test(datahub, model, config, evaluator, writer=None):
    '''evaluator must be loaded with correct codebook, db_codes and db_targets'''
    logging.info("compute mAP")
    model = model.to(config.device)
    test_mAP = evaluator.MAP(datahub.test_loader, model, topK=config.topK)
    logging.info("test mAP=%f" % test_mAP)
    logging.info("finish testing")
