---
title: 'def collate_fn(batch):    # ('

---

collate_fn 函數，用於在使用 DataLoader 時處理批次資料（batch data）。DataLoader 是 PyTorch 用於批次化訓練的工具，而 collate_fn 可以控制如何將單筆資料打包成一個批次，並且解決不同長度序列的填充問題（padding）

```python=
def collate_fn(batch):
    # (1) 提取 batch 中的輸入和標籤，並轉為 PyTorch tensor
    batch_x = [torch.tensor(data[0]) for data in batch]
    batch_y = [torch.tensor(data[1]) for data in batch]

    # (2) 計算每個序列的長度
    batch_x_lens = torch.LongTensor([len(x) for x in batch_x])  # 計算每個輸入序列長度
    batch_y_lens = torch.LongTensor([len(y) for y in batch_y])  # 計算每個標籤序列長度

    # (3) 將輸入序列填充 (Padding) 至相同長度
    pad_batch_x = torch.nn.utils.rnn.pad_sequence(
        batch_x, batch_first=True, padding_value=char_to_id['<pad>']
    )

    # (4) 將標籤序列填充至相同長度
    pad_batch_y = torch.nn.utils.rnn.pad_sequence(
        batch_y, batch_first=True, padding_value=char_to_id['<pad>']
    )

    # (5) 返回填充後的輸入、標籤及其長度
    return pad_batch_x, pad_batch_y, batch_x_lens, batch_y_lens

```


ds_train / ds_eval：是 Dataset，定義和存儲資料的集合。
dl_train / dl_eval：是 DataLoader，負責將 Dataset 分批並準備好給模型使用。


