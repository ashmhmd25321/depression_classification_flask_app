document.addEventListener("DOMContentLoaded", function () {
  console.log('login checker')
  if (getCookie('user') == '') {
    window.location = "login.php";
  }
  if (getCookie('type') == '') {
    window.location = "login.php";

  } else if (getCookie('type') == 1) {

  } else if (getCookie('type') == 2) {

    $('#Dashboard').css('display', 'none')
    $('#ManageTeacher').css('display', 'none')
    $('#manageSubject').css('display', 'none')
    $('#StudentDashboard').css('display', 'none')
  } else if (getCookie('type') == 3) {
    $('#Dashboard').css('display', 'none')
    $('#ManageTeacher').css('display', 'none')
    $('#ManageStudent').css('display', 'none')
    $('#manageSubject').css('display', 'none')
    $('#managePapers').css('display', 'none')
    $('#manageQuestion').css('display', 'none')
    $('#managePassPapers').css('display', 'none')
    $('#adminArea').css('display', 'none')
    $('#examArea').css('display', 'none')
  }
});

function setCookie(cname, cvalue) {
  const d = new Date();
  d.setTime(d.getTime() + (1 * 24 * 60 * 60 * 1000));
  let expires = "expires=" + d.toUTCString();
  document.cookie = cname + "=" + cvalue + ";" + expires + ";path=/";
}

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

function deleteCookie(cname) {
  const d = new Date();
  d.setTime(d.getTime() + (0));
  let expires = "expires=" + d.toUTCString();
  document.cookie = cname + "=" + '' + ";" + expires + ";path=/";
}

function Logout() {

  deleteCookie('user', '');
  deleteCookie('type', '');

}
