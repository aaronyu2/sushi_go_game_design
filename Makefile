.PHONY: all check_dir

all: check_dir

check_dir:
	@if [ ! -d "collected_data" ]; then \
		echo "Creating 'collected_data' directory..."; \
		mkdir -p collected_data; \
		echo 1 > "collected_data/filename"; \
	fi

TestSushi:
	@echo "#!/bin/bash" > TestSushi
	@echo "python3 test_sushi.py \"\$$@\"" >> TestSushi
	@chmod u+x TestSushi

clean:
	@rm -f TestSushi