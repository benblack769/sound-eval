function sum(v){
    var sum = 0;
    for(var i = 0; i < v.length; i++){
        sum += v[i]
    }
    return sum;
}
function mul_vecs(v1,v2){
    var res = new Array(v1.length)
    for(var i = 0; i < v1.length; i++){
        res[i] = v1[i] * v2[i];
    }
    return res;
}
function add_vecs(v1,v2){
    var res = new Array(v1.length)
    for(var i = 0; i < v1.length; i++){
        res[i] = v1[i] + v2[i];
    }
    return res;
}
function neg_vec(v){
    var res = new Array(v.length)
    for(var i = 0; i < v.length; i++){
        res[i] = -v[i];
    }
    return res;
}
function sub_vecs(v1,v2){
    return add_vecs(v1,neg_vec(v2))
}
function cosine_d(v1,v2){
    console.assert(v1.length == v2.length && v2.length > 0, "distance requires same length arrays")
    return 1.0 - sum(mul_vecs(v1,v2)) / (Math.sqrt(sum(mul_vecs(v1,v1))) * Math.sqrt(sum(mul_vecs(v2,v2))))
}
function closest(all_vecs, target, distance_fn){
    console.assert(all_vecs.length > 2, "closest only works when number of vectors is at least 2")
    var closest_idx = 0;
    var closest_dist = 10e20;
    for(var i = 0; i < all_vecs.length; i++){
        var this_dist = distance_fn(all_vecs[i],target);
        if(this_dist < closest_dist){
            closest_dist = this_dist;
            closest_idx = i;
        }
    }
    return closest_idx;
}
