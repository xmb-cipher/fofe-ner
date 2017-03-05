
var schema = {
    entity_types: [ 
        {
            type   : 'PER',
            labels : ['PER'],
            bgColor: '#FFCCAA',
            borderColor: 'darken'   },
        {
            type   : 'ORG',
            labels : ['ORG'],
            bgColor: '#8FB2FF',
            borderColor: 'darken'   },
        {
            type   : 'LOC',
            labels : ['LOC'],
            bgColor: '#95DFFF',
            borderColor: 'darken'   },
        {
            type   : 'MISC',
            labels : ['MISC'],
            bgColor: '#F1F447',
            borderColor: 'darken'   }, ]
};




// var docData = {
//     text     : "Ed O'Kelley was the man who shot the man who shot Jesse James.",
//     entities : [
//         ['T1', 'Person', [[0, 11]]],
//         ['T2', 'Person', [[20, 23]]],
//         ['T3', 'Person', [[37, 40]]],
//         ['T4', 'Person', [[50, 61]]],
//     ],
// };



head.ready(function() {
    // Util.embed('analysis', $.extend({}, collData),
    //         $.extend({}, docData), webFontURLs);
    $('#submit').click(function(){
        var userInput = $('#user-input').val();
        $.ajax({
            url: '/',
            type: 'POST',
            data: userInput,
            dateType: 'JSON',
            success: function(response) {
                console.log(response);
                $('#analysis').html('<div id="fofe-ner-out"></div>')
                Util.embed('fofe-ner-out', schema, response, webFontURLs);
            },
            error: function(error) {
                console.log(error);
            }
        });
    });
});