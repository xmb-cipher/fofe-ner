var bratLocation = 'static/js/'; //'https://storage.googleapis.com/corenlp/js/brat';
head.js(
    // External libraries
    bratLocation + '/client/lib/jquery.min.js',
    bratLocation + '/client/lib/jquery.svg.min.js',
    bratLocation + '/client/lib/jquery.svgdom.min.js',

    // brat helper modules
    bratLocation + '/client/src/configuration.js',
    bratLocation + '/client/src/util.js',
    bratLocation + '/client/src/annotation_log.js',
    bratLocation + '/client/lib/webfont.js',

    // brat modules
    bratLocation + '/client/src/dispatcher.js',
    bratLocation + '/client/src/url_monitor.js',
    bratLocation + '/client/src/visualizer.js'
);


var webFontURLs = [
    'static/fonts/Astloch-Bold.ttf',
    'static/fonts/PT_Sans-Caption-Web-Regular.ttf',
    'static/fonts/Liberation_Sans-Regular.ttf',
];

