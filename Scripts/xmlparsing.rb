require 'Nokogiri'
require 'json'

@f = File.open("brishti.xml")
@aggregation_uri='http://sead-test/fakeUri/0489a707-d428-4db4-8ce0-1ace548bc653_Aggregation'
@doc = Nokogiri::XML(@f)
@aggregation_uri = @doc.xpath("//rdf:Description[ore:describes/@rdf:resource='#{@aggregation_uri}']/*[starts-with(name(),'dcterms:')]")

@key=Array.new
@value=Array.new
for s in @aggregation_uri
  if s.to_s() =~ /<.*:(.*?)\s.*">\n\s*<.*>(.*)<\/.*>\n\s*.*\n\s*<\/.*>/ then
    a = ($1).to_s()
    b = ($2).to_s()
  elsif s.to_s() =~ /<.*?:(.*)\s.*>(.*)<\/.*>/ and $1.to_s != "" then
    a = ($1).to_s()
    b = ($2).to_s()
  elsif s.to_s() =~ /<.*?:(.*)\s*.*>(.*)<\/.*>/ and $1.to_s != "" then
    a = ($1).to_s()
    b = ($2).to_s()
  end

  @key.push(a)
  @value.push(b)
end

@len = @key.length - 1
@jsn=String.new
for i in 0..@len 
  @jsn = {"key" => ("dc."+ @key[i]).to_s(),"value" => @value[i].to_s(), "language" => "en"}
  puts @jsn
end

puts jsn