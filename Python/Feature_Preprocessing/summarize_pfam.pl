#!/usr/bin/perl -w

foreach my $result (@ARGV) {

	open PFAM, "$result";

	my $ctrl = 0;
	my $id = "";
	my %gpf;
	my @pfam = ();
	my $check_ctrl = 0;

	while (<PFAM>) {
		chomp;
		next if (/^#/);
		if (/^Query:\s+(\S+)\s+\[L=\d+\]/) {
			print "$id\t@pfam\n" if ($#pfam >= 0);
			$id = $1;
			$ctrl = 1;
			@pfam = ();
		}
		if (/Internal pipeline statistics summary/) {
			$ctrl = 0;
		}
		if (/No hits detected that satisfy reporting thresholds/) {
			print "$id\tno_hit\n";
			$ctrl = 0;
		}
		next if ($ctrl == 0);
		my $info = "";
		if (/^>>\s+(\S+)\s+/) {
			$model = $1;
			my $nextline = <PFAM>;
			if ($nextline =~ /No individual domains that satisfy reporting thresholds/) {
				$check_ctrl = 0;
			}
			else {
				$info = <PFAM>;
				$check_ctrl = 1;
			}
		}
		next unless ($check_ctrl == 1);
		while ($info = <PFAM>) {
			if ($info =~ /Alignments for each domain/) {
				$check_ctrl = 0;
				last;
			}
			elsif ($info =~ /No individual domains that satisfy reporting thresholds/) {
				$check_ctrl = 0;
				last;
			}
			my $push_ctrl = 1;
			my @arr = split(/\s+/,$info);
			next if ($#arr <= 0);
			my $model_span = "$model:$arr[10]-$arr[11]";
			if ($#pfam >= 0) {
				foreach (@pfam) {
					my @tmp = split(/:/,$_);
					my $span = pop @tmp;
					my ($start,$end) = split(/-/,$span);
					if ($arr[10] >= $start && $arr[10] <= $end) {
						$push_ctrl = 0;
						last;
					}
					elsif ($arr[11] >= $start && $arr[11] <= $end) {
						$push_ctrl = 0;
						last;
					}
				}
			}
			push (@pfam, $model_span) if ($push_ctrl == 1);
		}
	}
	print "$id\t@pfam\n" if ($#pfam >= 0);
	close PFAM;
}
