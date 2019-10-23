Code for CIKM paper Discerning Edge Influence for Network Embedding



### Requirements
> python 3.x  
> numpy

### Usage

```
python near.py -graph ../data/PPI.edgelist -embedding ../data/PPI.embedding -output PPI_near.embedding
```

```
-graph         graph data in edgelist format
-embedding     network embedding learned by skip-gram model
-output        embedding output by Near
```
