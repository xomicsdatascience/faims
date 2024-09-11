const charge_list = ['2', '3', '4', '5'];
const valid_char_list = ['F', 'a', 'E', 'T', 'M', 'm', 'R', 'V', 'A', 'K', 'I', 'G', 'W', 'P', 'Q', 'D', 'C', 'N', 'L', 'S', 'Y', 'H'];
const min_length = 7;
const max_length = 16;

function populate_example(elId){
    const el = document.getElementById(elId);
    el.textContent = "";
    const numEl = getRandomInt(3, 10);
    for(let i=0; i<numEl; i++){
        let chargeStr = charge_list[getRandomInt(0, charge_list.length)];
        let r = chargeStr + getRandomCharString();
        if(el.textContent.length !== 0){
            el.textContent += "\n";
        }
        el.textContent += r;
    }
}

function getRandomCharString(){
    let charString = "";
    for(let i=0; i<getRandomInt(min_length, max_length); i++){
        charString += valid_char_list[getRandomInt(0, valid_char_list.length)]
    }
    return charString
}


function getRandomInt(minVal, maxVal) {
  return Math.floor(minVal) + Math.floor(Math.random() * (maxVal-minVal));
}