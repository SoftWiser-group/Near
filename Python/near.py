# Author: Yaojing Wang
import numpy as np
import math
import random
import time
import sys
import argparse

SIGMOID_BOUND = 8
SIGMOID_TABLE_SIZE = 10000

sigmoid_table = [0]*(SIGMOID_TABLE_SIZE+1)
embedding_size = 0
node_size = 0

def loadGraph(graph):
    links, weights = [], []
    with open(graph, 'r') as f:
        for l in f:
            l = l.strip().split(' ')
            links.append((int(l[0]),int(l[1])))
            if len(l) < 2:
                weights.append(1.0)
            else:
                weights.append(float(l[2]))
    return links, weights

def loadEmbedding(embedding):
    global embedding_size, node_size
    embed = []
    with open(embedding, 'r') as f:
        l = f.readline().strip().split(' ')
        node_size, embedding_size = int(l[0]), int(l[1])
        embed = [[]] * node_size
        for l in f:
            l = l.strip().split(' ')
            n_id = int(l[0])
            embed[n_id] = [float(i) for i in l[1:]]
    return embed

def initSigmoidTable():
    global sigmoid_table
    for k in range(SIGMOID_TABLE_SIZE+1):
        x = 2 * SIGMOID_BOUND * k / SIGMOID_TABLE_SIZE - SIGMOID_BOUND
        sigmoid_table[k] = 1 / (1 + math.exp(-x))

def fastSigmoid(e1, e2):
    x = np.dot(e1, e2)
    if x > SIGMOID_BOUND:
        return 1
    elif x < -SIGMOID_BOUND:
        return 0
    k = int((x + SIGMOID_BOUND) * SIGMOID_TABLE_SIZE / SIGMOID_BOUND / 2)
    return sigmoid_table[k]

def Near(args, emb, links, weights, dalink, daweights, daflag):
    Hi_g_losses = np.zeros(node_size * embedding_size)
    
    v1, v2 = dalink[0], dalink[1]
    vlist = [v1, v2]
    loss = daweights * (fastSigmoid(emb[v1],emb[v2]) - 1)
    g_loss = np.zeros(node_size * embedding_size)
    for i in range(embedding_size):
        g_loss[v1*embedding_size + i] = loss * emb[v2][i]
        g_loss[v2*embedding_size + i] = loss * emb[v1][i]

    Hi_g_loss = g_loss
    ave_Hi_g_loss = np.zeros(node_size * embedding_size)
    iters = 0
    while True:
        sample_id = int(random.random()*len(links))
        sv1 = links[sample_id][0]
        sv2 = links[sample_id][1]
        if sv1 not in vlist or sv2 not in vlist:
            continue
        sample_loss = weights[sample_id] * fastSigmoid(emb[sv1],emb[sv2]) * (1 - fastSigmoid(emb[sv1], emb[sv2]))
        sample_H_g_loss = np.zeros(node_size * embedding_size)

        sv1_w2 = np.dot(emb[sv2], Hi_g_loss[sv1 * embedding_size:(sv1+1) * embedding_size])
        sv2_w1 = np.dot(emb[sv1], Hi_g_loss[sv2 * embedding_size:(sv2+1) * embedding_size])
        if sv1 in vlist:
            for i in range(embedding_size):
                sample_H_g_loss[sv2 * embedding_size + i] = sample_loss * emb[sv1][i] * sv1_w2
                sample_H_g_loss[sv1 * embedding_size + i] = sample_loss * emb[sv2][i] * sv1_w2
        if sv2 in vlist:
            for i in range(embedding_size):
                sample_H_g_loss[sv2 * embedding_size + i] = sample_loss * emb[sv1][i] * sv2_w1
                sample_H_g_loss[sv1 * embedding_size + i] = sample_loss * emb[sv2][i] * sv2_w1

        Hi_g_loss = g_loss + Hi_g_loss - sample_H_g_loss
        iters += 1
        if iters % 20 == 0 and iters > 1000:
            ave_Hi_g_loss += Hi_g_loss
        if iters >= 2000:
            break

    ave_Hi_g_loss /= (len(links)*(iters-1000)/20)

    with open(args.output, 'w') as f:
        f.write('{} {}\n'.format(node_size, embedding_size))
        for i in range(node_size):
            f.write('{} {}\n'.format(i, ' '.join([str(i) for i in (np.array(emb[i])+daflag*ave_Hi_g_loss[i*embedding_size:(i+1)*embedding_size]).tolist()])))
        print('Saved!')


def main(args):
    initSigmoidTable()

    links, weights = loadGraph(args.graph)
    emb = loadEmbedding(args.embedding)

    da_id = int(random.random()*node_size)
    daflag = args.da

    Near(args, emb, links, weights, links[da_id], weights[da_id], -1)
    print(links[da_id])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='inputs')

    parser.add_argument('-graph', nargs='?', required=False, help='')
    parser.add_argument('-embedding', nargs='?', required=False, help='')
    parser.add_argument('-da', nargs='?', default=-1, required=False, help='')
    parser.add_argument('-output', nargs='?', required=False, help='')
    

    args = parser.parse_args()

    main(args)