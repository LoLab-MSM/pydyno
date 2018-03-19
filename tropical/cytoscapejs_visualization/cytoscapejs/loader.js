if (window['cytoscape'] === undefined) {
    console.log('starting loading');
    requirejs.config({

        paths: {
            'cytoscape': 'https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.2.10/cytoscape',
            'cytoscape-popper': 'https://cdn.rawgit.com/cytoscape/cytoscape.js-popper/3ad50859/cytoscape-popper',
            'popper': 'https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.14.1/umd/popper.min',
            'tippy': 'https://cdnjs.cloudflare.com/ajax/libs/tippy.js/2.3.0/tippy.min'
        },
        shim: {
            'cytoscape-popper':{
                deps: ['cytoscape', 'popper']
            }
        }
    });
    window.$ = window.jQuery = require('jquery');
    requirejs(['cytoscape', 'cytoscape-popper', 'popper', 'tippy'],
        function (cytoscape, cypopper, popper, tippy) {
            console.log('Loading Cytoscape.js Module...');
            window['popper'] = popper;
            window['tippy'] = tippy;
            cypopper(cytoscape);
            window['cytoscape'] = cytoscape;

            var event = document.createEvent("HTMLEvents");
            event.initEvent("load_cytoscape", true, false);
            window.dispatchEvent(event);

    });

}