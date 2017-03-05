
var collData = {
    entity_types: [ {
            type   : 'Person',
            labels : ['Person', 'Per'],
            bgColor: '#7fa2ff',
            borderColor: 'darken'
    } ]
};

var docData = {
    text     : "Ed O'Kelley was the man who shot the man who shot Jesse James.",
    // The entities entry holds all entity annotations
    entities : [
        /* Format: [${ID}, ${TYPE}, [[${START}, ${END}]]]
            note that range of the offsets are [${START},${END}) */
        ['T1', 'Person', [[0, 11]]],
        ['T2', 'Person', [[20, 23]]],
        ['T3', 'Person', [[37, 40]]],
        ['T4', 'Person', [[50, 61]]],
    ],
};


head.ready(function() {
    Util.embed('analysis', $.extend({}, collData),
            $.extend({}, docData), webFontURLs);
});