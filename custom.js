updateWearablePreviewer = function(iframe_id, options) {
  const iframe = document.getElementById(iframe_id);
  const event = { type: 'update', payload : { options } };
  iframe.contentWindow.postMessage(event, '*');
}

getBodyShapeURN = function(gender) {
  return ;
}

getWearableExampleOptions = function() {
  const gender = document.getElementById('gender-switcher').value;
  const wearableId = document.getElementById('wearable-category').value;
  const bodyShape = `urn:decentraland:off-chain:base-avatars:Base${gender}`;
  const profile = 'default';
  if (wearableId == "-None-") {
    return {
      bodyShape,
      profile
    };
  } else {
    const [contractAddress, itemId] = wearableId.split('/');
    return {
      contractAddress,
      itemId,
      bodyShape,
      profile
    };
  }
}

poseWearable = function(iframe_id) {
  const emotes = [
    'clap',
    'dab',
    'dance',
    'fashion',
    'fashion-2',
    'fashion-3',
    'fashion-4',
    'love',
    // 'money', // we used later
    'fist-pump',
    'head-explode'
  ];
  const number = Math.floor(Math.random() * emotes.length);
  const options = getWearableExampleOptions();
  options.emote = emotes[number];
  updateWearablePreviewer(iframe_id, options);
}

setBodyShape = function(iframe_id, button) {
  const gender = button.value == 'Female' ? 'Male' : 'Female';
  button.value = gender;
  const options = getWearableExampleOptions();
  updateWearablePreviewer(iframe_id, options);
}


updateWearableExample = function(iframe_id, select) {
  const options = getWearableExampleOptions();
  updateWearablePreviewer(iframe_id, options);
}

//document.addEventListener("DOMContentLoaded",function(){
//  console.log('DOMContentLoaded');
//});

