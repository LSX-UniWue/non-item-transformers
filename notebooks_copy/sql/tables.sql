CREATE TABLE `bibtex` (
  `content_id` int(10) unsigned NOT NULL default '0',
  `journal` text,
  `volume` varchar(255) default NULL,
  `chapter` varchar(255) default NULL,
  `edition` varchar(255) default NULL,
  `month` varchar(45) default NULL,
  `day` varchar(45) default NULL,
  `booktitle` text,
  `howPublished` varchar(255) default NULL,
  `institution` varchar(255) default NULL,
  `organization` varchar(255) default NULL,
  `publisher` varchar(255) default NULL,
  `address` varchar(255) default NULL,
  `school` varchar(255) default NULL,
  `series` varchar(255) default NULL,
  `bibtexKey` varchar(255) default NULL,
  `url` text,
  `type` varchar(255) default NULL,
  `description` text,
  `annote` varchar(255) default NULL,
  `note` text,
  `pages` varchar(50) default NULL,
  `bKey` varchar(255) default NULL,
  `number` varchar(45) default NULL,
  `crossref` varchar(255) default NULL,
  `misc` text,
  `bibtexAbstract` text,
  `simhash0` char(32) NOT NULL default '',
  `simhash1` char(32) NOT NULL default '',
  `simhash2` char(32) NOT NULL default '',
  `entrytype` varchar(30) default NULL,
  `title` text,
  `author` text,
  `editor` text,
  `year` varchar(45) default NULL
) DEFAULT CHARSET=utf8;

CREATE TABLE `bookmark` (
  `content_id` int(10) unsigned NOT NULL default '0',
  `url_hash` varchar(32) NOT NULL default '',
  `url` text NOT NULL,
  `description` text NOT NULL,
  `extended` text,
  `date` datetime NOT NULL default '1815-12-10 00:00:00'
) DEFAULT CHARSET=utf8;

CREATE TABLE `tas` (
  `user` int(10) unsigned NOT NULL default '0',
  `tag` varchar(255) character set utf8 collate utf8_bin NOT NULL default '',
  `content_id` int(10) unsigned NOT NULL default '0',
  `content_type` tinyint(1) unsigned default NULL,
  `date` datetime NOT NULL default '1815-12-10 00:00:00',
  KEY `content_id_idx` (`content_id`),
  KEY `user_idx` (`user`)
) DEFAULT CHARSET=utf8;

CREATE TABLE `tagtagrelations` (
  `user` int(10) unsigned NOT NULL default '0',
  `subtag` varchar(255) character set utf8 collate utf8_bin NOT NULL default '',
  `supertag` varchar(255) character set utf8 collate utf8_bin NOT NULL default '',
  `date` datetime NOT NULL default '1815-12-10 00:00:00',
  UNIQUE KEY `user` (`user`,`subtag`(150),`supertag`(150))
) DEFAULT CHARSET=utf8;

LOAD DATA LOCAL INFILE '/tmp/bibtex' INTO TABLE bibtex CHARACTER SET utf8;
LOAD DATA LOCAL INFILE '/tmp/bookmark' INTO TABLE bookmark CHARACTER SET utf8;
LOAD DATA LOCAL INFILE '/tmp/tas' INTO TABLE tas CHARACTER SET utf8;
LOAD DATA LOCAL INFILE '/tmp/relation' INTO TABLE tagtagrelations CHARACTER SET utf8;



