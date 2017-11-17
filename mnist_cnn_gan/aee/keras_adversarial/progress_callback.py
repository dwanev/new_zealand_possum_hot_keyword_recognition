from keras.callbacks import Callback

from .image_grid import write_image_grid


class ProgressCallback(Callback):
    def __init__(self):
        pass

    def get_last_value_from_dict(self,dict,key):
        if (key in dict):
            a = dict[key]
            if (len(a)>0):
                return(a[len(a) - 1])
            else:
                return(None)
        else:
            return(None)

    def on_epoch_end(self, epoch, logs={}):
        print('ProgressCallback - epoch end')
        for key in self.model.history.history.keys():
            print(key)

        val_loss = self.get_last_value_from_dict(self.model.history.history,'val_loss')
        print('epoch',epoch,'val_loss',val_loss)
        val_acc = self.get_last_value_from_dict(self.model.history.history,'val_acc')
        print('epoch',epoch,'val_acc',val_acc)


