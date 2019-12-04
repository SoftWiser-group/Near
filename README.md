Code for CIKM paper Discerning Edge Influence for Network Embedding



### Requirements
> python 3.x  
> numpy

### Usage

```
python near.py -graph ../data/PPI.edgelist -embedding ../data/PPI.embedding -output PPI_near.embedding -da -1 -edge 123
```

```
-graph         graph data in edgelist format
-embedding     network embedding learned by skip-gram model
-da            select to delete or add edge, 0: random select edge(default), 1:add, -1:delete
-edge          edge id in edgelist (start from 0)
-output        embedding output by Near
```
### Citing
```
@inproceedings{wang2019discerning,
  title={Discerning Edge Influence for Network Embedding},
  author={Wang, Yaojing and Yao, Yuan and Tong, Hanghang and Xu, Feng and Lu, Jian},
  booktitle={Proceedings of the 28th ACM International Conference on Information and Knowledge Management},
  pages={429--438},
  year={2019},
  organization={ACM}
}
```
