    function submit() {
        var filename = $("#iname").val();
        if(filename){
            $.ajax({
                data: {
                    filename : filename,
                },
                type : 'POST',
                url : '/search'
            })
                .done(function(data){
                    $("#imgdiv").empty();
                    var main = document.getElementById("imgdiv");
                    

                    for(var i = 0; i < data.images.length; ++i) {
                        if( i % 3==0 )
                        {
                            var row = document.createElement("div");
                            row.className = "row";
                            main.appendChild(row);
                        }
                            
                        var image = " <img src=\"/static/image/".concat(data.images[i], "\" height=\"500px\" width=\"250px\"> ");
                        var col = '<div class="column m-4"/> ' + image + "<br> <p>" + data.images[i] + "</p>" +' <div/>'
                        row.innerHTML = row.innerHTML +  col;
                    }
                });
            event.preventDefault();
        }
    }
