[![Build Status](https://travis-ci.org/LoLab-VU/tropical.svg?branch=master)](https://travis-ci.org/LoLab-VU/tropical)
[![Coverage Status](https://coveralls.io/repos/github/LoLab-VU/tropical/badge.svg?branch=master)](https://coveralls.io/github/LoLab-VU/tropical?branch=master)
[![Code Health](https://landscape.io/github/LoLab-VU/tropical/master/landscape.svg?style=flat)](https://landscape.io/github/LoLab-VU/tropical/master)

# TroPy

We present TroPy, a novel approach to study cellular signaling networks from a dynamic perspective. This method combines the Quasi-Steady State approach with max-plus algebra to find the driver species and the specific reactions that contribute the most to species concentration changes in time. Hence, it is possible to study how those driver species and reactions change for different parameter sets that fit the data equally well. Finally, it is possible to identify clusters of parameter that generate similar modes of signal execution in signal transduction pathways.

## Running TroPy



```python
# Importing libraries
import pickle
from tropical import clustering
```


```python
# Loading dynamic signatures
with open('signatures.pickle', 'rb') as handle:
    all_signatures = pickle.load(handle)

# Obtaining signature of how the species 0 (the enzyme) is being consumend
sp0_sign_reactants = all_signatures[0]['consumption']

# Initializing clustering object
clus = clustering.ClusterSequences(sp0_sign_reactants, unique_sequences=False)
clus.diss_matrix()
clus.silhouette_score_spectral_range(4)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cluster_silhouette</th>
      <th>num_clusters</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.976426</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.714087</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.799800</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
%matplotlib inline
# Clustering into two groups as 2 has the best Silhouette score
clus.spectral_clustering(2)
# Plotting clustered signatures
pl = clustering.PlotSequences(clus)
pl.all_trajectories_plot()
```


![png](double_enzymatic_analysis_files/double_enzymatic_analysis_2_0.png)



```python
clus_info = [{0:clus.cluster_percentage_color(), 'best_silh':clus.silhouette_score()}]
clus_info
```




    [{0: {0: (0.8679245283018868,
        '#000000',
        array([ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                0.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,
                2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,
                2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,
                2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,
                2.,  2.,  2.,  2.,  2.,  2.,  2.,  1.,  1.,  1.,  1.,  1.,  1.,
                1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.])),
       1: (0.1320754716981132,
        '#FFFF00',
        array([ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
                1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
                1.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,
                2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,
                2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,
                2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,
                2.,  2.,  2.,  2.,  2.,  2.,  2.,  0.,  0.,  0.,  0.,  0.,  0.,
                0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]))},
      'best_silh': 0.97642585995189124}]




```python
from tropical.cytoscapejs_visualization.model_visualization import ModelVisualization
from tropical.examples.double_enzymatic.mm_two_paths_model import model

viz = ModelVisualization(model)
data_viz = viz.static_view(get_passengers=True, cluster_info=clus_info)

```


```python
from tropical.cytoscapejs_visualization.cytoscapejs import viewer as cyjs
from IPython.display import display
q=cyjs.render(data_viz)
```


    <IPython.core.display.Javascript object>



<!DOCTYPE html>
<html>
<head>
    <meta charset=utf-8 />
    <link href="http://cdnjs.cloudflare.com/ajax/libs/qtip2/2.2.0/jquery.qtip.min.css" rel="stylesheet" type="text/css" />
    <style type="text/css">
        body {
            font: 14px helvetica neue, helvetica, arial, sans-serif;
        }

        ul li {
            background: #ffe5e5;
            padding: 5px;
        }

        .tooltiptext{
            display: none;
        }

        .input-color {
            position: relative;
        }
        .input-color input {
            padding-left: 20px;
        }
        .input-color .color-box {
            width: 10px;
            height: 10px;
            display: inline-block;
            background-color: #ccc;
            position: absolute;
            left: 5px;
            top: 8px;
        }

        #cy82140b75-53a6-4ab4-96eb-3deaa6d456cc {
            height: 700px;
            width: 100%;
            position: absolute;
            left: 4px;
            top: 5px;
            background: #FFFFFF;
        }
        #play42803bbd-8db6-498a-a53a-ae09291c4d22 {
            width: 5%;
            height: 4%;
            font-size:130%;
            position: absolute;
            top: 93%;
            left: 10%;
        }

        #reset6d5d76bf-4b7d-4b79-ad12-3f101d8cfbef {
            width: 7%;
            height: 4%;
            font-size:130%;
            position: absolute;
            top: 93%;
            left: 16%;
        }

        #range76bd7664-e209-44ae-8245-7d5002eff468 {
            width:80%;
            height:25px;
            top:89%;
            left:10%;
            border:1px solid black;
            position:absolute;
        }

        #text47533582-c09f-4c95-9adf-3c0cdf3951c5 {
            width:10%;
            height:5%;
            top:92.5%;
            left:24%;
            font-size:120%;
            border:0px solid black;
            position:absolute;
        }
        #fit3ccd9160-35e8-4143-8e8d-37643e506bff {
            width: 3em;
            margin: 0.5em;
            position: absolute;
            z-index: 999;
            top: 0;
            right: 0;
        }

        #fit3ccd9160-35e8-4143-8e8d-37643e506bff i {
            transform: rotate(45deg);
        }

    </style>

    <script>
        (function() {
            var all_data ={"elements": {"nodes": [{"position": {"y": 72.0, "x": 27.0}, "data": {"clus_scores": [0.9764258599518912, 0.9764258599518912], "clus_perc": [86.79245283018868, 13.20754716981132], "clus_colors": ["#000000", "#FFFF00"], "label": "E", "clus_repres": [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], "background_color": "#2b913a", "id": "s0", "name": "s0"}}, {"position": {"y": 126.0, "x": 27.0}, "data": {"background_color": "#162899", "id": "s1", "name": "s1", "label": "S1"}}, {"position": {"y": 18.0, "x": 27.0}, "data": {"background_color": "#162899", "id": "s2", "name": "s2", "label": "S2"}}, {"position": {"y": 99.0, "x": 119.27}, "data": {"background_color": "#162899", "id": "s3", "name": "s3", "label": "E_S1"}}, {"position": {"y": 45.0, "x": 119.27}, "data": {"background_color": "#162899", "id": "s4", "name": "s4", "label": "E_S2"}}, {"position": {"y": 72.0, "x": 211.54}, "data": {"background_color": "#2b913a", "id": "s5", "name": "s5", "label": "P"}}], "edges": [{"data": {"source_arrow_shape": "diamond", "target_arrow_shape": "triangle", "source": "s0", "target": "s3"}}, {"data": {"source_arrow_shape": "diamond", "target_arrow_shape": "triangle", "source": "s0", "target": "s4"}}, {"data": {"source_arrow_shape": "diamond", "target_arrow_shape": "triangle", "source": "s1", "target": "s3"}}, {"data": {"source_arrow_shape": "diamond", "target_arrow_shape": "triangle", "source": "s2", "target": "s4"}}, {"data": {"source_arrow_shape": "none", "target_arrow_shape": "triangle", "source": "s3", "target": "s0"}}, {"data": {"source_arrow_shape": "none", "target_arrow_shape": "triangle", "source": "s3", "target": "s5"}}, {"data": {"source_arrow_shape": "none", "target_arrow_shape": "triangle", "source": "s4", "target": "s0"}}, {"data": {"source_arrow_shape": "none", "target_arrow_shape": "triangle", "source": "s4", "target": "s5"}}]}, "data": {"name": "tropical.examples.double_enzymatic.mm_two_paths_model", "view": "static"}};
            function render() {

                var cy = window.cy = cytoscape({
                    container: $('#cy82140b75-53a6-4ab4-96eb-3deaa6d456cc'),
                    style: cytoscape.stylesheet()
                        .selector('node')
                        .style({
                            'label': 'data(label)',
                            'pie-size': '80%',
                            'pie-1-background-color': 'data(background_color)',
                            'pie-1-background-size': '100',
                            'pie-2-background-color': '#dddcd4',
                            'pie-2-background-size': '100'
                        })

                        .selector('edge')
                        .style({
                            'curve-style': 'bezier',
                            // 'width': 'data(width)',
                            'target-arrow-shape': 'data(target_arrow_shape)',
                            'source-arrow-shape': 'data(source_arrow_shape)'
                        }),
                    elements: all_data.elements,

                    layout: {name: 'preset'
                    },

                    boxSelectionEnabled: true
                });

                var dynamics_vis = function(){
                    // setting qtip style
                    cy.elements().qtip({
                        content: ' ',
                        position: {
                            my: 'bottom right',
                            at: 'bottom left'
                        },
                        style: {
                            classes: 'qtip-bootstrap',
                            tip: {
                                width: 8,
                                height: 4
                            }
                        }
                    });
                    var tspan = all_data.data.tspan;
                    var text = $('#text47533582-c09f-4c95-9adf-3c0cdf3951c5')[0];
                    var rangeInput = $('#range76bd7664-e209-44ae-8245-7d5002eff468')[0];
                    rangeInput.max = tspan.length;


                    function update_nodes(t){
                        rangeInput.value = t;
                        text.value = tspan[t].toFixed(2);
                        cy.batch(function(){
                            cy.edges().forEach(function (e) {
                                var c = e.data('edge_color')[t];
                                var s = e.data('edge_size')[t];
                                var a = e.data('edge_qtip')[t];
                                e.style({'line-color': c,
                                    'target-arrow-color': c,
                                    'source-arrow-color': c,
                                    'width': s});

                                // e.animate({
                                //     style: {'line-color': c,
                                //         'target-arrow-color': c,
                                //         'source-arrow-color': c,
                                //         'width': s},
                                //     duration: 100,
                                //     queue: false
                                // });
                                e.qtip('api').set('content.text', a.toString());
                                // n.animate({style: {'width': s}})

                            });
                            cy.nodes().forEach(function(n){
                                var p = n.data('rel_value')[t];
                                var q = n.data('abs_value')[t];
                                n.style({'pie-1-background-size': p});

                                // n.animate({
                                //     style: {'pie-1-background-size': p},
                                //     duration: 100,
                                //     queue: false
                                // });

                                n.qtip('api').set('content.text', q.toString())
                            });
                        })
                    }

                    var currentTime = 0;
                    var tInterval = null;

                    function nextTime(){
                        currentTime = currentTime+1;
                        if (currentTime >= tspan.length){
                            clearInterval(tInterval)
                        }
                        else {update_nodes(currentTime)}
                    }

                    var playing = false;
                    var playButton = $('#play42803bbd-8db6-498a-a53a-ae09291c4d22')[0];

                    function pauseSlideshow(){
                        playButton.innerHTML = 'Play';
                        playing = false;
                        clearInterval(tInterval);
                    }

                    function playSlideshow(){
                        playButton.innerHTML = 'Pause';
                        playing = true;
                        tInterval = setInterval(nextTime, 1000)
                    }


                    var resetButton = $('#reset6d5d76bf-4b7d-4b79-ad12-3f101d8cfbef')[0];
                    resetButton.onclick = function(){
                        pauseSlideshow();
                        currentTime = 0;
                        update_nodes(currentTime)
                    };

                    playButton.onclick = function(){
                        if(playing){ pauseSlideshow(); }
                        else{ playSlideshow(); }
                    };
                    rangeInput.addEventListener('mouseup', function(){
                        pauseSlideshow();
                        currentTime = this.value;
                        currentTime = parseInt(currentTime);
                        update_nodes(currentTime);

                    });

                    rangeInput.onchange = function(){
                        text.value = tspan[this.value].toFixed(2);
                        // currentTime = this.value
                    };
                };

                var cluster_info = function () {
                    cy.nodes("[clus_colors]").qtip({
                        content: ' ',
                        position: {
                            viewport: $(window)
                        },
                        style: {
                            classes: 'qtip-bootstrap',
                            tip: {
                                width: 800,
                                height: 4
                            }
                        }
                    });
                    var divClone = $('#clus60070f82-e19e-4962-92bd-fc531077f4c7').clone();
                    cy.batch(function () {
                        cy.nodes("[clus_colors]").forEach(function (n) {
                            var bg = n.data('clus_colors');
                            var perc = n.data('clus_perc');
                            var scores = n.data('clus_scores');
                            var repres = n.data('clus_repres');
                            var i;
                            var pie_bg = {};
                            for (i = 0; i < bg.length; i++){
                                pie_bg['pie-'+(i+1)+'-background-color'] = bg[i];
                                pie_bg['pie-'+(i+1)+'-background-size'] = perc[i];
                                var data = "";
                                data +="<li style='list-style: none;'>";
                                data +="<div class='input-color'> <input type='text' " +
                                    "value='"+ perc[i].toFixed(2) +" "+'%,'+" "+ repres[i] +"' /> <div " +
                                    "class='color-box' style='background-color:"+ bg[i] +";' </div></div>";
                                data +="</li>";
                                $('#clus60070f82-e19e-4962-92bd-fc531077f4c7').append(data) ;

                            }
                            n.style(pie_bg);
                            n.qtip('api').set('content.text', $('#clus60070f82-e19e-4962-92bd-fc531077f4c7'));
                            n.qtip('api').set('content.title', n.data('label')+', score: ' +scores[0].toFixed(2));
                            $('#clus60070f82-e19e-4962-92bd-fc531077f4c7').replaceWith(divClone.clone());

                        })

                    });
                };

                var static_vis = function () {
                    $('#text47533582-c09f-4c95-9adf-3c0cdf3951c5')[0].style.display = 'none';
                    $('#range76bd7664-e209-44ae-8245-7d5002eff468')[0].style.display = 'none';
                    $('#play42803bbd-8db6-498a-a53a-ae09291c4d22')[0].style.display = 'none';
                    $('#reset6d5d76bf-4b7d-4b79-ad12-3f101d8cfbef')[0].style.display = 'none';
                };

                if(all_data.data.view == 'dynamic'){
                    dynamics_vis();
                } else {
                    static_vis();
                    cluster_info()
                }

                var allNodes;
                var layoutPadding = 50;
                var aniDur = 500;
                var easing = 'linear';
                allNodes = cy.nodes();
                $('#fit3ccd9160-35e8-4143-8e8d-37643e506bff').on('click', function(){

                    allNodes.unselect();
                    cy.stop();

                    cy.animation({
                        fit: {
                            eles: cy.elements(),
                            padding: layoutPadding
                        },
                        duration: aniDur,
                        easing: easing
                    }).play();

                });

            }

            var before_render = function(){
                if(window['cytoscape'] === undefined){
                    console.log("Waiting for Cyjs...");
                    window.addEventListener("load_cytoscape", before_render);
                } else {
                    console.log("Ready to render graph!");
                    render();
                }
            };

            before_render();

        })();
    </script>
</head>

<body>
<div id="cy82140b75-53a6-4ab4-96eb-3deaa6d456cc"></div>
<!-- When only #uuid div is placed on this page,
the height of output-box on ipynb will be 0px.
One line below will prevent that. -->
<button class="controls" id=play42803bbd-8db6-498a-a53a-ae09291c4d22>Play</button>
<button class="controls" id=reset6d5d76bf-4b7d-4b79-ad12-3f101d8cfbef>Reset</button>
<input type="range" class="slider" id=range76bd7664-e209-44ae-8245-7d5002eff468 min="0" max="50" value="0" step="1">
<input type="text" size="400" id=text47533582-c09f-4c95-9adf-3c0cdf3951c5 value="0">
<button id=fit3ccd9160-35e8-4143-8e8d-37643e506bff class="btn btn-default"><i class="fa fa-arrows-h"></i></button>
<div id="dummy" style="width:100px;height:700px"></div>
<div class="tooltiptext" id="clus60070f82-e19e-4962-92bd-fc531077f4c7"></div>
</body>

</html>



```python
%matplotlib inline
from tropical.analysis_of_clusters import AnalysisCluster

ac = AnalysisCluster(model=model, sim_results='sim_results.h5', clusters=clus.labels)
ac.plot_dynamics_cluster_types(species=[0], norm=False)
```


![png](double_enzymatic_analysis_files/double_enzymatic_analysis_6_0.png)



![png](double_enzymatic_analysis_files/double_enzymatic_analysis_6_1.png)

