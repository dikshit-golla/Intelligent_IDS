document.addEventListener('DOMContentLoaded', function(){
  const form = document.getElementById('kdd-form');
  form.addEventListener('submit', function(e){
    // Basic client-side check: ensure required numeric fields are filled
    const required = form.querySelectorAll('[required]');
    for(let el of required){
      if(!el.value){
        e.preventDefault();
        alert('Please fill all required fields.');
        el.focus();
        return;
      }
    }
  });
});
