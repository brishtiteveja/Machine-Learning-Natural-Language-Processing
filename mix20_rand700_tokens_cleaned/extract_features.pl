use Clair::Document;
my $dataset_dir = shift;
my $output_file = shift;
my @categories = `ls $train_dir`;
my %features=();
$i=1;
open(OUT,">$output_file");
foreach $cat(@categories){
  chomp($cat);
    @files = `ls  $train_dir/$cat`;
  foreach $file (@files){
          chomp($file);
          print OUT $cat, " ";
          my $text = `cat $train_dir/$cat/$file`;
          $text =~ s/|//g;
          my $doc = Clair::Document->new(string => $text);
          my %tf = $doc->tf(type => "stem");
          while ( my ($key, $value) = each(%tf) ) {
             my $k;
             if(defined $features{$key}){
                $k=$features{$key};
             }else{
                $k=$i;
                $features{$key}=$i;
                $i++;
             }
             print OUT "$k:$value ";
          }
          print OUT "\n";
     }
}
