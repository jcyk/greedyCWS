# greedyCWS

Hi, this code is easy to use!

Please check the `src/train.py` for all hyper-parameter and IO settings.

You can modify the `src/train.py` to speficy your own model settings or datasets.


- For training, use the command line `python train.py`. Training details will be printed on the screen. The learned parameters will be saved in in the same directory as `train.py` *per epoch*, which will be named as `epoch1`, `epoch2`, `...`.
- For test, the ***same*** command line `python train.py` is used, but with a specified parameter file (e.g., `epoch1`), via the function argument `load_params` in `train.py` (Note `load_params` should be `None` when training). In addition, tell your test file by setting `dev_file` (Yes, when test, consider it as "test_file"). The segmented result will be saved in `src/result`.

The code is originally designed for reasearch purpose, but adaptable to industrial use. 


## Citation
This code implements an efficient and effective neural word segmenter proposed in the following paper.

Deng Cai, Hai Zhao, etc., Fast and Accurate Neural Word Segmentation for Chinese. ACL 2017.

If you find it useful, please cite the [paper](http://aclweb.org/anthology/P17-2096).
```
@InProceedings{cai-EtAl:2017:Short,
  author    = {Cai, Deng  and  Zhao, Hai  and  Zhang, Zhisong  and  Xin, Yuan  and  Wu, Yongjian  and  Huang, Feiyue},
  title     = {Fast and Accurate Neural Word Segmentation for Chinese},
  booktitle = {Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)},
  month     = {July},
  year      = {2017},
  address   = {Vancouver, Canada},
  publisher = {Association for Computational Linguistics},
  pages     = {608--615},
  url       = {http://aclweb.org/anthology/P17-2096}
}
```

## Contact
Drop me (Deng Cai) an email at thisisjcykcd (AT) gmail.com if you have any question.


