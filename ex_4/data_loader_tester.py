from gcommand_loader import GCommandLoader
import torch
import soundfile

if __name__ == '__main__':

    dataset = GCommandLoader('train')

    test_loader = torch.utils.data.DataLoader(
            dataset, batch_size=100, shuffle=None,
             num_workers=20, pin_memory=True, sampler=None)

    for k, (input,label) in enumerate(test_loader):
        print(input.size(), len(label))
