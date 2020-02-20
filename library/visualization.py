from matplotlib import pyplot as plt


colors = ['crimson', 'mediumpurple', 'steelblue', 'forestgreen', 'khaki', 'orange', 'salmon', 'lightgreen', 'deeppink', 'navy', 'teal']

def plot_statistics(model_statistics_time, topic_statistics, plotname=None, titles=None):
    '''
    This method visualizes model statistics 

    '''
    plt.rc('font', size=25)

    if titles is None:
        titles = [
            'Семантическая неоднородность от n_t',
            'Бинарная семантическая неоднородность от n_t',
            'Семантическая неоднородность с бинарной функцией потерь от n_t',
            'Бинарная семантическая неоднородность с бинарной функцией потерь от n_t',
            'Семантическая загрязнённость от n_t',
            'Бинарная семантическая загрязнённость от n_t',
            'Семантическая загрязнённость с бинарной функцией потерь от n_t',
            'Бинарная семантическая загрязнённость с бинарной функцией потерь от n_t'
        ]

    num_subplots = len(topic_statistics)
    num_statistics = len(titles)

    fig = plt.figure(figsize=(16 * num_statistics, 8 * num_subplots))

    for ind, (tau, stat) in enumerate(topic_statistics.items()):
        for i in range(num_statistics):
            plt.subplot(num_subplots, num_statistics, ind * num_statistics + 1 + i)
            plt.scatter(model_statistics_time[tau][1].nt, stat[i], c=colors[ind])

            plt.xlabel('n_t')
            plt.ylabel(f'tau={tau}')

            if not ind:
                plt.title(titles[i])

            if i:
                plt.ylim(-0.05, 1.05)
            else:
                plt.ylim(-0.05, max(stat[0]) + 0.1)

    fig.tight_layout()

    if plotname is not None:
        plt.savefig(plotname)


def plot_top_tokens_evolution():
    pass
