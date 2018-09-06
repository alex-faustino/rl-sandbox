
all: websync

websync:
	rsync -Pprl web/ sftp.courses.engr.illinois.edu:/courses/me598mw/fa2018
