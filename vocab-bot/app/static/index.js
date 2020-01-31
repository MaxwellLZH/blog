var app = new Vue({
	el: "#app",
	data: {
	}
})


var app2 = new Vue({
	el: "#w",
	data: {
		words: []
	}
})


// Make a request for a user with a given ID
axios.get('/backend/tofel/get_words')
  .then(function (response) {
    console.log(response);
    app2.words = response.data
  })
  .catch(function (error) {
    // handle error
    console.log(error);
  })
  .then(function () {
    // always executed
  });


