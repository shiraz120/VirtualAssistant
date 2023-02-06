# VirtualAssistant
VirtualAssistant with nlp
That project is in develop, it will implement the main model with PyTorch and manage each microservice with Docker. also the frontend will be implemented with Dart using Flutter.
The way the virtual assistant is going to work is by receiving some request from the user, clean it and pass it to a NER model and regular expressions that will 
extract the entities from the request (if there are any) and replace them with tags that are known by the main model.
For instance, given the request "send an email to example@gmail.com" it will replace it to "send an email to EMAIL_0" and the tag with it's corresponding email will be 
kept in a dictionary.
what the main model is going to do is receive that request, analyze it and return a parsable program that contain which web service the request is meant for and 
what are the parameters to pass to that web service. 
for the example given above the main model might output something like "notify => @com.gmail ( dest = EMAIL_0, message = "" )", the web service is Gmail and the parameters are the destiny email and the message to send to it. then the parameter EMAIL_0 from the output
will be replaced with "example@gmail.com" and the program will be passed to a parser that will extract the parameters and the correct web service and that is going
to be passed to a container that will request that web service with those parameters and return an answer to the user.
