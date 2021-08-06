import dot2tex
import graphviz
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .stats import ResultsPartiallyConfounded, ResultsFullyConfounded


def _pval_to_str(pval, alpha=0.05, floor=0.0001):
    if pval is None:
        return ''
    elif pval > alpha:
        return 'p=' + str(np.round(pval, 2))
    elif pval > floor:
        return 'p=' + str(np.round(pval, np.ceil(-np.log10(pval)).astype(int)))
    else:
        return 'p<' + str(floor)


def plot_null_dist(confound_test_results, **kwargs):
    if not hasattr(confound_test_results, 'null_distribution'):
        raise RuntimeError("No null distribution data is available. "
                           "Re-run 'confound_test' with 'return_null_dist=True'!")
    g = sns.histplot(confound_test_results.null_distribution, **kwargs)
    g.set(xlabel='R2(y^,c*)', ylabel='count')
    g.set_title('null distribution')
    plt.axvline(confound_test_results.r2_yhat_c, color='red')
    return g


def plot_graph(confound_test_results, y_name='y', yhat_name='<y&#770;>', c_name='c', outfile_base=None, precision=3):
    if isinstance(confound_test_results, ResultsPartiallyConfounded):
        mode = 'partial'
    else:
        mode = 'full'

    return plot_r2_graph(confound_test_results.r2_y_c,
                         confound_test_results.r2_yhat_c,
                         confound_test_results.r2_y_yhat,
                         confound_test_results.p,
                         y_name=y_name, yhat_name=yhat_name, c_name=c_name,
                         mode=mode,
                         outfile_base=outfile_base,
                         precision=precision)


def plot_r2_graph(r2_y_c, r2_yhat_c, r2_y_yhat, p_yhat_c=None,
                  y_name='y', yhat_name='yhat', c_name='c',
                  mode='partial',
                  outfile_base=None,
                  precision=3,
                  alpha=0.05,
                  minp=0.0001):
    dot = graphviz.Graph()
    dot.attr(rankdir='LR')

    if mode != 'partial' and mode != 'full':
        raise AttributeError("Mode must be either 'partial' or 'full'.")

    if p_yhat_c < alpha:
        star = '*'
    else:
        star = ''
    pvalstr = ' (' + _pval_to_str(p_yhat_c, alpha, minp) + star + ')'

    dot.node('c', label=c_name)
    dot.node('y', label=y_name)
    dot.node('yhat', label=yhat_name)

    if mode == 'partial':
        dot.edge('c', 'yhat', label=str(np.round(r2_yhat_c, precision)) + pvalstr, style="dashed")
    else:
        dot.edge('c', 'yhat', label=str(np.round(r2_yhat_c, precision)))
    dot.edge('c', 'y', label=str(np.round(r2_y_c, precision)))
    if mode == 'full':
        dot.edge('y', 'yhat', label=str(np.round(r2_y_yhat, precision)) + pvalstr, style="dashed")
    else:
        dot.edge('y', 'yhat', label=str(np.round(r2_y_yhat, precision)))

    if outfile_base is not None:
        dot.render(filename=outfile_base + '.dot')  # saves dot and pdf
        tex_code = dot2tex.dot2tex(dot.source, format='tikz', crop=True)
        with open(outfile_base + '.tex', "w") as text_file:
            text_file.write(tex_code)

    return dot
