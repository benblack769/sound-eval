var input_json_data = null;
var filename_set = null;
var song_list = null;
var numpy_vecs = null;

function make_dropdown_menu(base_sel,add_distance){
    if(input_json_data.length == 0){
        return;
    }
    var object = Object.assign({},input_json_data[0]);
    delete object.x;
    delete object.y;
    delete object.filename;
    delete object.citation;

    var keys = Object.keys(object);
    if(add_distance){
        keys.push("distance_to_selected")
    }
    base_sel.innerHTML = ""
    add_to_dropdown_menu(base_sel,keys)
}
function add_to_dropdown_menu(parent_element, choices_list){
    for(var i = 0; i < choices_list.length; i++) {
        var opt = document.createElement('option');
        opt.innerHTML = choices_list[i];
        opt.value = choices_list[i];
        parent_element.appendChild(opt);
    }
}
function downloadURI(uri, name) {
    var link = document.createElement("a");
    link.download = name;
    link.href = uri;
    link.type = "audio/mid"
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    delete link;
}
function dis_to_label(dist){
    if(dist < 0.7){
        return "<"+Math.ceil(dist*10)/10
    }
    else{
        return ">0.7"
    }
}
function select_element(filename){
    document.getElementById("selected_display").innerText = filename
    set_audio_with_display()
    if(numpy_vecs){
        var idx = song_list.indexOf(filename)
        var sel_vector = numpy_vecs[idx]
        for(var i = 0; i < numpy_vecs.length; i++){
            var dist = cosine_d(numpy_vecs[i],sel_vector)
            input_json_data[i].distance_to_selected = dis_to_label(dist)
        }
        make_graphic()
    }
}
function list_key_values(data){
    var str = ""
    for(key in data){
        if(key != "x" && key != "y" && key != "filename" && key != "citation"){
            str += key + ":" + data[key] + ",  "
        }
    }
    return str
}
function make_graphic(){
    var selected = $("#category_options").val();
    var filtered_input_json = fliter_out_selected()
	var graphic_args = {
		title: "Musica",
		description: "Click to copy point filenames to ",
		width: 800,
		height: 600,
		data:filtered_input_json,
		target: "#data_plot",
		x_accessor: "x",
		y_accessor: "y",
		color_accessor: selected,
		color_type:'category',
		chart_type:'point',
		//legend: ['arg','var'],
		click_to_zoom_out: false,
		brush: 'xy',
		click: function(d){
            select_element(d.data.filename)
		},
        mouseover: function(d, i) {
            // custom format the rollover text, show days
            d3.select('#data_plot svg .mg-active-datapoint')
              .text(list_key_values(d.data));
        },
    }
    function to_color(n){
        if(n == ">0.7"){
            return "#ffffff"
        }
        else{
            var num = parseInt(n[3])
            var code = ((num-1)*2).toString(16)
            return  "#"+code.repeat(6)
        }
    }
    if(selected == "distance_to_selected"){
        graphic_args.color_domain = ["<0.1", "<0.2", "<0.3", "<0.4", "<0.5", "<0.6", "<0.7", ">0.7"]
        graphic_args.color_range = graphic_args.color_domain.map(to_color)
    }
    MG.data_graphic(graphic_args);

	$("#zoom_out_button").click(function(){
		MG.zoom_to_raw_range(graphic_args)
	})
}
function set_audio(filename,citation){
    document.getElementById("audio_source").src = filename
    document.getElementById("citation").innerText = citation
    document.getElementById("audio_id").load()
}
function set_audio_with_display(){
    var this_val = document.getElementById("selected_display").innerText;
    if(filename_set.has(this_val)) {
        var idx = song_list.indexOf(this_val)
        var dict_data = input_json_data[idx]
        var citation = "citation" in dict_data ? input_json_data[idx]['citation'] : "";
        set_audio("mp3_files/"+this_val,citation)
    }
}
function setup_interactive(){
    var filename_list = input_json_data.map(dict=>dict['filename'])

    $("#category_options").change(make_graphic)

    $("#select_options").change(change_selection)
    $("#select_out_options").change(make_graphic)
}
function fliter_out_selected(){
    var key = $("#select_options").val()
    var value = $("#select_out_options").val()

    var filtered = input_json_data.filter(dict=>dict[key]==value)
    return value == "All" ? input_json_data : filtered;
}
function change_selection(){
    var sel_val = $("#select_options").val()
    choices = Array.from(new Set(input_json_data.map(dict=>dict[sel_val])))
    choices.unshift("All")
    var parent = document.getElementById("select_out_options")
    parent.innerHTML = ""
    add_to_dropdown_menu(parent,choices)
}

function load_data(){
    $.getJSON("all_data.json",function(json){
        input_json_data = json;
        filename_set = new Set(input_json_data.map(dict=>dict.filename))
        song_list = input_json_data.map(dict=>dict.filename)
        make_dropdown_menu(document.getElementById("category_options"),numpy_vecs)
        make_dropdown_menu(document.getElementById("select_options"),numpy_vecs)
        setup_interactive()
        change_selection()
        make_graphic()
    })
	$.getJSON("vec_json.json",function(json){
		numpy_vecs = json;
        make_dropdown_menu(document.getElementById("category_options"),numpy_vecs)
		$(".vec_calc_elmt").show();
	})
}

$(document).ready(function(e) {
    load_data()
})
