package utils

import (
	"fmt"
	"regexp"
	"strings"
)

type CommentRecognize struct {
	sensitiveKeys        map[string]int
	sensitiveCombineKeys [][]string
	protectKeys          map[string]int
	protectCombineKeys   [][]string
	phoneNumberReg       *regexp.Regexp
	idReg                *regexp.Regexp
}

func (c *CommentRecognize) Init() {
	c.sensitiveCombineKeys = [][]string{{"job", "money"},
		{"job", "month"},
		{"job", "day"},
		{"job", "daily"},
		{"job", "apply"},
		{"per", "day"},
		{"job", "income"},
		{"work", "income"},
		{"$", "month"},
		{"$", "daily"},
		{"money", "daily"},
		{"money", "month"},
		{"refer", "id"},
		{"refer", "code"},
		{"google", "play"},
		{"donwload", "http"},
		{"onead"},
		{"pay", "app"},
		{"pay", "job"},
		{"pay", "company"},
		{"bonus"},
		{"helpline"},
		{"store", "app"},
		{"helpline"},
		{"earn", "month"},
		{"earn", "day"},
		{"earn", "$"},
		{"earn", "job"},
		{"call", "get"},
		{"business", "whatsapp"},
		{"part", "time", "job"},
		{"vmate", "uninstall"},
		{"ragistration", "investment"},
		{"registration", "investment"},
		{"registe", "investment"},
		{"requir", "boy"},
		{"requir", "girl"},
		{"डाउनलोड", "कमायें"},   //下载，赚
		{"कमाई"},                //收益
		{"कमायें", "रुपया"},     //赚，卢比
		{"कमायें", "हज़ार ाशि"}, //赚，千卢比
		{"रेफर", "कोड"}}         //推荐码
	c.protectCombineKeys = [][]string{{"vmate", "official"}}

	c.phoneNumberReg = regexp.MustCompile("\\d{2}-\\d{2}-\\d{2}-\\d{2}-\\d{2}|\\d{2}-\\d{3}-\\d{3}-\\d{2}")
	c.idReg = regexp.MustCompile("(\\d+?)")

	c.sensitiveKeys = map[string]int{}
	for _, values := range c.sensitiveCombineKeys {
		for _, value := range values {
			_, err := c.sensitiveKeys[value]
			if !err {
				c.sensitiveKeys[value] = 1
			}
		}
	}
	c.protectKeys = map[string]int{}
	for _, values := range c.protectCombineKeys {
		for _, value := range values {
			_, err := c.protectKeys[value]
			if !err {
				c.protectKeys[value] = 1
			}
		}
	}
}

func (c *CommentRecognize) Recognize(comment string) int {
	// 输入评论返回识别结果 0:无影响评论；1:广告评估
	result := c.AdRecognize(comment)
	if result {
		return 1
	}
	return 0
}

func (c *CommentRecognize) AdRecognize(comment string) bool {
	// 输入评论返回是否广告评估
	comment = strings.ToLower(comment)
	protectResult := c.countProtect(comment)
	sensitiveResult := c.countSensitive(comment)
	id := c.likeNumber(comment)
	if sumMap(protectResult) >= 1 {
		return false
	}
	ns := sumMap(sensitiveResult)
	if (id && ns >= 1) || (ns >= 2) {
		return true
	}
	return false

}

func (c *CommentRecognize) countSensitive(comment string) map[string]int {
	var combinekeysResult map[string]int
	combinekeysResult = c.countKeys(comment, c.sensitiveCombineKeys, c.sensitiveKeys)
	return combinekeysResult
}

func (c *CommentRecognize) countProtect(comment string) map[string]int {
	var combinekeysResult map[string]int
	combinekeysResult = c.countKeys(comment, c.protectCombineKeys, c.protectKeys)
	return combinekeysResult
}

// 统计各种Keys组合的数目（每一个组合是一个特征）
func (c *CommentRecognize) countKeys(comment string, combineKeys [][]string, keys map[string]int) map[string]int {
	var keysResult map[string]int
	var combinekeysResult map[string]int
	keysResult = make(map[string]int)
	combinekeysResult = make(map[string]int)
	for word, _ := range keys {
		n := strings.Count(comment, word)
		if n >= 1 {
			keysResult[word] = 1
		} else {
			keysResult[word] = 0
		}
	}
	for _, words := range combineKeys {
		key := ""
		value := 1
		for i, word := range words {
			if i == 0 {
				key = word
			} else {
				key = fmt.Sprintf("%s_%s", key, word)
			}
			if keysResult[word] == 0 {
				value = 0
			}
		}
		combinekeysResult[key] = value
	}
	return combinekeysResult
}

// 判断是否像一个联系方式
func (c *CommentRecognize) likeNumber(comment string) bool {
	r1 := c.phoneNumberReg.MatchString(comment)
	if r1 {
		return true
	} else {
		n := len(strings.Split(comment, " "))
		m := len(c.idReg.FindAllString(comment, -1))
		if m >= 10 {
			if strings.Contains(comment, "whatsapp") || strings.Contains(comment, "phone") || n >= 5 {
				for i := 0; i < n-20; i++ {
					if len(c.idReg.FindAllString(comment[i:i+20], -1)) >= 10 {
						return true
					}
				}
			}
		}
	}
	return false
}

func sumMap(maps map[string]int) int {
	n := 0
	for _, v := range maps {
		n += v
	}
	return n
}
