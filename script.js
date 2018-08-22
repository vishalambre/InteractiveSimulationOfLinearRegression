let x_val=[];
let y_val = [];

let y_predicted=[];
const learningRate = 0.02;
const optimizer = tf.train.sgd(learningRate);


let m,b; //For line equation y = mx + b 

function loss(predictedLabel,label){
    tlabel = tf.tensor1d(label);  // label receives y_val, which needs to be convereted to a tensor
    tloss = predictedLabel.sub(tlabel).square().mean();
    // console.log('tloss');
    // tloss.print();
    return tloss;
}


function predict(xs){

    txs = tf.tensor1d(xs);
    tys = txs.mul(m).add(b);
    // console.log('tys');
    // tys.print();
    return tys;

}

function setup(){
    createCanvas(400, 400);
    
    m = tf.variable(tf.scalar(random(1)));
    b = tf.variable(tf.scalar(random(1)));

}

function mousePressed(){
   

    // console.log(`MouseX is ${mouseX} MouseY is ${mouseY}`);

    //  Mapping needs to be done
    xt = map(mouseX,0,width,0,1);
    yt = map(mouseY,0,height,0,1);

    x_val.push(xt);
    y_val.push(yt);
    
}

function draw(){


    tf.tidy(() => {

        if(x_val.length>0){
            optimizer.minimize(()=>{
                return loss(predict(x_val),y_val);
            })
        }
        
        background(0);
    
        strokeWeight(8);
        stroke(255);
        for(let i=0;i<x_val.length;i++){
            ptx = map(x_val[i],0, 1, 0, width);
            pty = map(y_val[i],0, 1, 0, height);
            point(ptx,pty);
        }
    
        
    
        strokeWeight(4);
        let x= [0,1]
        x1 = map(x[0],0,1,0,width);
        x2 = map(x[1],0,1,0,width);
    
        ty = predict(x);
        ys = ty.dataSync();
    
        y1 = map(ys[0],0,1,height,0); // 1 and 0 positiion are exchanged  to get a line from left bottom to top right
        y2 = map(ys[1],0,1,height,0); 
    
        line(x1,y1,x2,y2);
    

    });
    
    // console.log(tf.memory().numTensors);
}