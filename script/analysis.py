import numpy as np

from utils import cprint, get_carray, plt, set_colorbar


def filter_refcomp(df, sort='True', compact=True):
    dref = df.loc[df['conductivity'].notna(), :]
    if sort:
        dref.sort_values(by='conductivity', ascending=False, inplace=True)
    if compact:
        dref = dref[['material_id', 'formula', 'band_gap', 'e_hull', 
                 'conductivity', 'lattice_type', 'space_group', 'Y', 'Z']]
    return dref


def violin_plot(ax, dataset, palette='Paired', colors=None, half=False, vert=False,
                showmeans=False, showmedians=False, showextrema=False, 
                simplify=False, specialc=None, specialc_loc=None, alpha=1,
                title=None, xlabel=None, ylabel=None, yticklabels=None, fs=12):
    _dataset = list(map(lambda x: np.squeeze(x), dataset))  # flatten each data
    if colors is None:
        colors = get_carray(len(_dataset), palette=palette,
                            specialc=specialc, specialc_loc=specialc_loc)
    
    violin = ax.violinplot(_dataset, points=500, showmeans=showmeans, 
                           showmedians=showmedians, showextrema=showextrema, 
                           vert=vert)
    for idx, b in enumerate(violin['bodies']):
        if half:
            b.get_paths()[0].vertices[:, 1] = \
                np.clip(b.get_paths()[0].vertices[:, 1], idx+1, idx+2)  # modify it so we only see the upper half of the violin plot
        b.set_color(colors[idx,:])  # change to the desired color
        b.set_alpha(alpha)
        b.set_facecolor(colors[idx,:])
        b.set_edgecolor(colors[idx,:])
        if simplify:
            values = np.mean(_dataset[idx])
            locations = idx+1
        else:
            values = _dataset[idx]
            if values.ndim == 0:
                locations = [idx+1]
            else:
                locations = np.repeat([idx+1], len(values))
        ax.scatter(values, locations, marker='o', color='white', 
                    edgecolor='k', s=3, linewidths=1, zorder=100)  # mark mean value
    
    # Decoration
    if title:
        ax.set_title(title, fontsize=fs)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=fs)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=fs)
    if yticklabels:
        ax.set_yticks(np.arange(1, len(dataset) + 1))
        ax.set_yticklabels(yticklabels, fontsize=fs)
    return violin


def raincloud_plot(ax, dataset, ylabels, xlabel, palette='Paired', alpha=0.4,
                   specialc=None, specialc_loc=None, simplify=False, 
                   simplify_labels=False):
    colors = get_carray(len(dataset), palette=palette,
                        specialc=specialc, specialc_loc=specialc_loc)
    
    # Box plot
    bp = ax.boxplot(dataset, patch_artist=True, vert=False, showfliers=False)
    for (patch, color) in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(alpha)
        
    # Violin plot
    violin_plot(ax, dataset, palette=palette, half=True, showmeans=False,
                specialc=specialc, specialc_loc=specialc_loc, simplify=simplify)
        
    # Scatter plot
    for idx, data in enumerate(dataset):
        y = np.full(len(data), idx + 0.8)  # add jitter so the features do not overlap on the y-axis
        idxs = np.arange(len(y))
        out = y.astype(float)
        out.flat[idxs] += np.random.uniform(low=-.05, high=.05, size=len(idxs))
        y = out
        ax.scatter(data, y, s=.3, c=colors[idx,:])
    
    if ylabels is None:
        ax.set_yticks(np.arange(1, len(dataset)+1, 1), [None]*len(dataset))
    else:
        if simplify_labels:
            ylabels = simplify_label(ylabels)
        ax.set_yticks(np.arange(1, len(dataset)+1, 1), ylabels)
    ax.set_xlabel(xlabel)
    return


def simplify_label(name):
    if isinstance(name, list):
        recursive = True
    else:
        recursive = False
        
    if recursive:
        name_ = list(map(lambda x: simplify_label(x), name))
    else:
        kind = name.split(' ')[0]
        if kind == 'Group':
            name_ = 'G' + name.split(' ')[1]
        elif kind == 'Promising':
            name_ = 'P' + name.split(' ')[1]
        else:
            name_ = name
    return name_


def locate_arrows(ax, name, x, y, arrow_locs, count, fs, arrowprops, 
                  display_count=False, simplify=True):
    if arrow_locs:
        try:
            arrow_loc = arrow_locs[name]
        except:
            arrow_loc = False
    else:
        arrow_loc = None
    
    if simplify:
        name = simplify_label(name)
    
    if display_count:
        string = name + ' (' + str(count) + ')'
    else:
        string = name
    
    if arrow_loc is False:
        x_, y_ = (0, 0)
        ax.annotate(string, (x, y), xytext=(x+x_, y+y_), fontsize=fs)
    elif arrow_loc is None:
        pass
    else:
        x_, y_ = arrow_loc
        ax.annotate(string, (x, y), xytext=(x+x_, y+y_), arrowprops=arrowprops, fontsize=fs)
    return


def plot_manifold(df, target='Y', remove_axis=False, remove_ticks=False, 
                  mag=False, margin=3, # general
                  group_properties=None, mapping='linear', palette='Paired', 
                  ordering=False, specialc=None, specialc_loc=None, alpha=1, # coloring
                  colorbar=False, shrink=0.9, labelpad=3, colorbar_label=None, # colorbar
                  overlay_data=False, log_transformation=False, data_unit=None, 
                  s=24, s_data=24, palette_data='winter',  # data overlay
                  orientation='horizontal', pos=[0.25, 0.9, 0.6, 0.03], 
                  arrow_locs=None, fs=10, simplify=True, verbose=True):  # arrow
    cprint('Plot data manifold...', color='w')
    
    # Choose samples
    df.reset_index(inplace=True, drop=True)
    Z = np.stack(df['Z'].values)
    Y = np.stack(df[target].values)
      
    # Scatter
    fig, ax = plt.subplots(1,1, figsize=(4,4))
    if group_properties:
        names = group_properties[0]
        indexer = {name: i for i, name in enumerate(names)}
        Y_ = [indexer[y] for y in Y]  # represented in numbers
        c = get_carray(Y_, palette=palette, specialc=specialc, specialc_loc=specialc_loc)
    else:
        c = get_carray(Y, palette=palette)
        
    if ordering:
        n = 5
        i = 0
        segment = np.linspace(min(Y), max(Y), n)
        for _a, _b in zip(segment[:-1], segment[1:]):
            indices = (_a <= Y) & (Y <= _b)
            c[:,-1] = alpha
            ax.scatter(Z[indices,0], Z[indices,1], c=c[indices,:], s=s/n*(i+1), 
                       marker='o', edgecolors='none')
            i += 1
    else:
        c[:,-1] = alpha
        ax.scatter(Z[:,0], Z[:,1], c=c, s=s, edgecolors='none')

    # Decoration
    if mag:
        ax.set_xlim((mag[0] - margin, mag[1] + margin))
        ax.set_ylim((mag[2] - margin, mag[3] + margin))
    else:
        x_lim = ax.get_xlim()
        y_lim = ax.get_ylim()
        ax.set_xlim((x_lim[0] - margin, x_lim[1] + margin))
        ax.set_ylim((y_lim[0] - margin, y_lim[1] + margin))
    if remove_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
    if remove_axis:
        ax.set_axis_off()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    ax.set_xlabel('PACMAP1', fontsize=fs)
    ax.set_ylabel('PACMAP2', fontsize=fs)
    
    # Overlay datapoints
    if overlay_data:
        dref = filter_refcomp(df, sort=False)
        if log_transformation:
            dref['log_' + overlay_data] = dref[overlay_data].apply(lambda x: np.log10(x))
            values = dref['log_' + overlay_data]
        else:
            values = dref[overlay_data]
        c = get_carray(values, palette=palette_data)
        
        Zref = np.stack(dref['Z'])
        ax.scatter(Zref[:,0], Zref[:,1], c=c, marker='s', s=s_data, edgecolors='k', linewidths=1)

        if colorbar:
            if orientation != 'horizontal':
                raise('#TODO')
            cbar = set_colorbar(fig, pos=pos, values=values, palette=palette_data, 
                                label=data_unit, orientation=orientation, 
                                labelpad=labelpad, shrink=shrink)
            cbar.ax.xaxis.set_ticks_position('bottom')
            cbar.ax.xaxis.set_label_position('top')
    
    if group_properties:
        names, counts, centers, band_gaps, e_hulls = group_properties
        # arrowprops = dict(arrowstyle='wedge', shrinkA=1, shrinkB=1,
        #                   connectionstyle='angle3,angleA=0,angleB=-90',
        #                   edgecolor='none', facecolor='k', alpha=0.7)
        arrowprops = dict(arrowstyle='-', shrinkA=1, shrinkB=1,
                          edgecolor='k', facecolor='k', alpha=0.7)
        for i, name in enumerate(names):
            count = counts[i]
            center = centers[i]
            band_gap = band_gaps[i]
            e_hull = e_hulls[i]
            x = center[0]
            y = center[1]
            locate_arrows(ax=ax, name=name, x=x, y=y, arrow_locs=arrow_locs, 
                          count=count, fs=fs, arrowprops=arrowprops, simplify=simplify)
            if verbose:
                cprint(name, '| center :', center, color='w')
    
    plt.subplots_adjust(right=0.875, left=0.2)
    return ax


def migrate_pack_data(pack, pack_, names, name):
    idx = names.index(name)
    pack_['names'].append(pack['names'][idx])
    pack_['counts'].append(pack['counts'][idx])
    pack_['centers'].append(pack['centers'][idx])
    pack_['band_gaps'].append(pack['band_gaps'][idx])
    pack_['e_hulls'].append(pack['e_hulls'][idx])
    return pack_


def organize_pack(pack):
    names, counts, centers, band_gaps, e_hulls = pack.values()
    pack_ = make_pack()
    n = 0
    if 'Noise' in names:  # noise data goes first
        pack_ = migrate_pack_data(pack, pack_, names, 'Noise')
        n += 1
    
    for prefix in ['Group', 'Promising']:
        try:  # 1, 2, 3, ...
            suffix = sorted([int(name.split(' ')[1]) for name in names if prefix in name])
        except:  # 2A, 2B, ...
            suffix = sorted([name.split(' ')[1] for name in names if prefix in name])
            
        for num in suffix:
            name = prefix + ' ' + str(num)
            pack_ = migrate_pack_data(pack, pack_, names, name)
            n += 1
    
    assert len(names) == n
    return pack_


def get_spec(ds):
    count = len(ds)
    pos = ds['Z'].mean()
    e_hull = ds['e_hull'].values
    band_gap = ds['band_gap'].values
    return count, pos, e_hull, band_gap


def packaging(pack, name, ds):
    count, pos, e_hull, band_gap = get_spec(ds)
    pack['names'].append(name)
    pack['counts'].append(count)
    pack['centers'].append(pos)
    pack['band_gaps'].append(band_gap)
    pack['e_hulls'].append(e_hull)
    return pack


def make_pack():
    return {'names': [], 'counts': [], 'centers': [], 'band_gaps': [], 'e_hulls': []}


def group_analysis(df, palette='Paired', alpha=1, specialc=None, 
                   specialc_loc=None, simplify=True, draw=False):
    cprint('Analyze groups...', color='w')
    uY = df['Y'].unique()
    
    # Grouping
    pack = make_pack()
    for y in sorted(uY):
        ds = df[df['Y'] == y]
        pack = packaging(pack, y, ds)
    pack = organize_pack(pack)
    
    # Unpacking
    names, counts, centers, band_gaps, e_hulls = pack.values()
    
    if draw:
        # Boxplot
        if len(uY) > 15:
            figsize = (6, 8)
        else:
            figsize = (6, 4)
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Raincloud plot
        raincloud_plot(ax=axes[0], dataset=band_gaps, ylabels=names, 
                    xlabel='Band gap [eV]', palette=palette, alpha=alpha,
                    specialc=specialc, specialc_loc=specialc_loc, simplify=simplify)
        raincloud_plot(ax=axes[1], dataset=e_hulls, ylabels=None, 
                    xlabel='$E_{hull}$ [eV/atom]', palette=palette, alpha=alpha,
                    specialc=specialc, specialc_loc=specialc_loc, simplify=simplify)
        
        # Decoration
        fig.subplots_adjust(wspace=0.15, left=0.2, bottom=0.17)
    return names, counts, centers, band_gaps, e_hulls