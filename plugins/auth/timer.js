const startingMinutes = getCookie('time');
let time = startingMinutes * 60;
const countDownEl = $('#timer');
let interval = setInterval(function () {


  const minutes = Math.floor(time / 60)
  let seconds = time % 60;
  seconds = seconds < 10 ? '0' + seconds : seconds;
  countDownEl.text(minutes + ':' + seconds)
  time--;
  if (minutes==0 && seconds==0){
    countDownEl.css('color','red')
    countDownEl.css('font-style','bold')
    clearInterval(interval);
  }
}, 1000);

function getCookie(cname) {
  let name = cname + "=";
  let decodedCookie = decodeURIComponent(document.cookie);
  let ca = decodedCookie.split(';');
  for (let i = 0; i < ca.length; i++) {
    let c = ca[i];
    while (c.charAt(0) == ' ') {
      c = c.substring(1);
    }
    if (c.indexOf(name) == 0) {
      return c.substring(name.length, c.length);
    }
  }
  return "";
}
