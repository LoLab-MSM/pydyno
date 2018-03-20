if (window['cytoscape'] === undefined) {
    console.log('starting loading');
    requirejs.config({

        paths: {
            'popper': 'https://unpkg.com/popper.js@1.14.1/dist/umd/popper',
            'tippy': 'https://cdnjs.cloudflare.com/ajax/libs/tippy.js/2.3.0/tippy.min',
            'dagre': 'https://unpkg.com/dagre@0.8.2/dist/dagre',
            'cola': 'https://unpkg.com/webcola/WebCola/cola.min',
            'cytoscape': 'https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.2.10/cytoscape',
            'cytoscape-popper': 'https://cdn.rawgit.com/cytoscape/cytoscape.js-popper/3ad50859/cytoscape-popper',
            'cytoscape-dagre': 'https://cdn.rawgit.com/cytoscape/cytoscape.js-dagre/7246e548/cytoscape-dagre',
            'cytoscape-euler': 'https://cdn.rawgit.com/cytoscape/cytoscape.js-euler/b5a6e24c/cytoscape-euler',
            'cytoscape-cose-bilkent': 'https://cdn.rawgit.com/cytoscape/cytoscape.js-cose-bilkent/d810281d/cytoscape-cose-bilkent',
            'cytoscape-cola': 'https://cdn.rawgit.com/cytoscape/cytoscape.js-cola/09008ecc/cytoscape-cola'
        },
        shim: {
            'cytoscape-popper':{
                deps: ['cytoscape', 'popper']
            },
            'cytoscape-dagre':{
                deps: ['cytoscape', 'dagre']
            },
            'cytoscape-euler':{
                deps: ['cytoscape']
            },
            'cytoscape-cose-bilkent':{
                deps: ['cytoscape']
            },
            'cytoscape-cola':{
                deps: ['cytoscape', 'cola']
            }
        },

        map: {
            '*': {
                'popper.js': 'popper',
                'webcola': 'cola'}
        }
    });
    window.$ = window.jQuery = require('jquery');
    requirejs(['cytoscape', 'cytoscape-popper', 'cytoscape-dagre', 'cytoscape-euler', 'cytoscape-cose-bilkent',
            'cytoscape-cola', 'popper',  'tippy', 'dagre', 'cola'],
        function (cytoscape, cypopper, cydagre, cyeuler, cybilkent, cycola, popper, tippy, dagre, cola) {
            console.log('Loading Cytoscape.js Module...');
            window['popper'] = popper;
            window['tippy'] = tippy;
            window['dagre'] = dagre;
            window['cola'] = cola;
            window['cytoscape'] = cytoscape;
            cypopper(cytoscape);
            cydagre(cytoscape);
            cyeuler(cytoscape);
            cybilkent(cytoscape);
            cycola(cytoscape);

            var event = document.createEvent("HTMLEvents");
            event.initEvent("load_cytoscape", true, false);
            window.dispatchEvent(event);

    });

}