# Font issue on linux/unix

In container environment, it is quite often to meet issue about font not properly displayed / extracted, e.g. from PDF.

It is recommended to install below font lib to allow the OS to read CJK characters.

```docker
# To support Japanese in PDF export - https://wiki.alpinelinux.org/wiki/Fonts
RUN apk add terminus-font font-noto font-noto-thai \
	font-noto-tibetan font-ipa font-sony-misc font-daewoo-misc font-jis-misc
```

Or,  another [source](https://github.com/Belval/pdf2image/issues/215) showing below font libs

```docker
RUN apt-get install fonts-arphic-ukai fonts-arphic-uming \
	fonts-ipafont-mincho fonts-ipafont-gothic fonts-unfonts-core
```