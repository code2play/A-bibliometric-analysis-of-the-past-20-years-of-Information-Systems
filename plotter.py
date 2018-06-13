import os

import numpy as np
import pandas as pd
from wordcloud import WordCloud
from matplotlib import pyplot as plt

font_style1 = {
    'family':'serif', 
    'size':16, 
    'weight':'bold'
}
font_style2 = {
    'family':'serif', 
    'size':12, 
    'weight':'bold'
}

def plot_corr_matrix(corr, Dir):
    labels = corr.index
    corr = corr.values
    n = corr.shape[0]

    value = []
    for i in range(n):
        for j in range(n):
            if i!=j:
                value.append(corr[i][j])
    # value = np.array(value)
    # mean = value.mean()
    # std = value.std()
    minv = min(value)
    maxv = max(value)
    for i in range(n):
        for j in range(n):
            if i!=j:
                corr[i][j] = (corr[i][j]-minv)/(maxv-minv)

    fig, ax = plt.subplots()
    fig.set_size_inches(10,8)
    im = ax.imshow(corr, cmap='summer')

    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(labels, font_style2)
    ax.set_yticks(np.arange(n))
    ax.set_yticklabels(labels, font_style2)
    plt.setp(ax.get_xticklabels(), rotation=-90, ha="left",
                rotation_mode="anchor")

    cbar = ax.figure.colorbar(im, ax=ax)
    cbar_label = cbar.get_ticks()
    cbar_label = ['{:.1f}'.format(l) for l in cbar_label]
    cbar.ax.set_yticklabels(cbar_label, font_style2)

    # for edge, spine in ax.spines.items():
    #     spine.set_visible(False)

    # ax.set_xticks(np.arange(n+1)-.5, minor=True)
    # ax.set_yticks(np.arange(n+1)-.5, minor=True)
    # ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    # ax.tick_params(which="minor", bottom=False, left=False)

    fig.tight_layout()
    plt.savefig(Dir+'co-presence.pdf', format='pdf')
    plt.close()
    # plt.show()


def plot_word_cloud(words, probs, Dir):
    if not os.path.exists(Dir+'word cloud'):
        os.mkdir(Dir+'word cloud')

    index = list(words.index)

    for i, j in enumerate(index):
        # print(i, j)
        word = list(words.iloc[i, :])
        # prob = list(probs.iloc[i, :])
        prob = range(1000, 500, -50)

        seq = {word[_]:prob[_] for _ in range(10)}
        wc = WordCloud(
            width=1000,
            height=1000,
            random_state=42,
            font_path='./fonts/SourceHanSansCN-Normal.otf',
            background_color='white',
            colormap='Blues_r'
                    # BuPu
                    # Purples_r
                    # RdBu_r
                    # RdYlBu
                    # Set2
                    # Spectral
                    # summer
                    # viridis
        ).fit_words(seq)
        plt.figure().set_size_inches(8, 8)
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(Dir+'word cloud/{}-{}.pdf'.format(i, j), format='pdf')
        plt.close()


def plot_confusion_matrix(cm, labels, Dir):
    n = len(labels)
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 8)
    im = ax.imshow(cm)

    plt.xlabel('y_pred', font_style1)
    plt.ylabel('y_true', font_style1)
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(labels, font_style1)
    ax.set_yticks(np.arange(n))
    ax.set_yticklabels(labels, font_style1)
    # ax.tick_params(top=True, bottom=False,
    #                labeltop=True, labelbottom=False)
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right",
            rotation_mode="anchor")
    # for i in range(n):
    #     for j in range(n):
    #         if cm[i][j]==0:
    #             continue
    #         text = ax.text(j, i, cm[i, j],
    #                     ha="center", va="center", color="w")
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar_label = cbar.get_ticks()
    cbar.ax.set_yticklabels(cbar_label, font_style1)
    plt.tight_layout()
    plt.savefig(Dir+'confusion matrix.pdf', format='pdf')
    # plt.show()


def plot_innovation_score(score, year):
    fig, ax = plt.subplots()
    ax.set_xticks(year)
    ax.set_xticklabels(year)
    plt.bar(year, score)
    plt.xlabel('Year')
    plt.ylabel('Innovation Score')
    plt.show()


if __name__=='__main__':
    # jrnl_topic_num = [50, 75, 100, 150, 200, 300]
    # conf_topic_num = [200, 300, 400, 500, 600, 700]

    # dirs = [
    #     './results/6.6/only keywords/{}/{} topics/',
    #     './results/6.6/with keywords/{}/{} topics/'
    # ]

    # for Dir in dirs:
    #     for n_topics in jrnl_topic_num:
    #         d = Dir.format('journal', n_topics)
    #         print(d)
    #         words = pd.read_csv(d+'topic terms.csv', 
    #                             index_col=0)
    #         probs = pd.read_csv(d+'topic terms probability.csv', 
    #                             index_col=0)
    #         plot_word_cloud(words, probs, d)
    #     for n_topics in conf_topic_num:
    #         d = Dir.format('conference', n_topics)
    #         print(d)
    #         words = pd.read_csv(d+'topic terms.csv', 
    #                             index_col=0)
    #         probs = pd.read_csv(d+'topic terms probability.csv', 
    #                             index_col=0)
    #         plot_word_cloud(words, probs, d)

    Dirs = [
        './results/6.6/only keywords/journal/100 topics/',
        './results/6.6/with keywords/conference/200 topics/'
    ]

    for Dir in Dirs:
        cm_df = pd.read_csv(Dir+'confusion matrix.csv', index_col=0)
        labels = list(cm_df.index)
        cm = cm_df.values
        plot_confusion_matrix(cm, labels, Dir)

        corr = pd.read_csv(Dir+'co-presence.csv', index_col=0)
        plot_corr_matrix(corr, Dir)