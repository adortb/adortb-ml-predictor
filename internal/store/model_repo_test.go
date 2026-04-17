package store

import (
	"encoding/json"
	"testing"
	"time"
)

func TestRepo_SaveAndGet(t *testing.T) {
	repo, err := NewRepo(t.TempDir())
	if err != nil {
		t.Fatalf("创建仓库失败: %v", err)
	}

	data, _ := json.Marshal(map[string]any{"weights": map[string]float64{"f1": 0.5}})
	meta := ModelMeta{
		Name:    "lr_ctr",
		Version: "v1",
		Type:    ModelTypeLR,
		Samples: 10000,
	}
	if err := repo.Save(meta, data); err != nil {
		t.Fatalf("保存失败: %v", err)
	}

	got, ok := repo.Get("v1")
	if !ok {
		t.Fatal("期望找到版本 v1")
	}
	if got.Version != "v1" || got.Type != ModelTypeLR {
		t.Errorf("元数据不匹配: %+v", got)
	}
}

func TestRepo_GetNotFound(t *testing.T) {
	repo, _ := NewRepo(t.TempDir())
	_, ok := repo.Get("nonexistent")
	if ok {
		t.Error("期望未找到，但返回 true")
	}
}

func TestRepo_Latest(t *testing.T) {
	repo, _ := NewRepo(t.TempDir())

	data := []byte(`{}`)
	now := time.Now()

	repo.Save(ModelMeta{Name: "lr_ctr", Version: "v1", Type: ModelTypeLR, CreatedAt: now.Add(-2 * time.Hour)}, data)
	repo.Save(ModelMeta{Name: "lr_ctr", Version: "v2", Type: ModelTypeLR, CreatedAt: now.Add(-1 * time.Hour)}, data)
	repo.Save(ModelMeta{Name: "lr_ctr", Version: "v3", Type: ModelTypeLR, CreatedAt: now}, data)

	latest, ok := repo.Latest(ModelTypeLR)
	if !ok {
		t.Fatal("期望有最新版本")
	}
	if latest.Version != "v3" {
		t.Errorf("最新版本应为 v3，得到 %s", latest.Version)
	}
}

func TestRepo_LatestEmptyRepo(t *testing.T) {
	repo, _ := NewRepo(t.TempDir())
	_, ok := repo.Latest(ModelTypeLR)
	if ok {
		t.Error("空仓库不应有最新版本")
	}
}

func TestRepo_List(t *testing.T) {
	repo, _ := NewRepo(t.TempDir())

	data := []byte(`{}`)
	for i, v := range []string{"v1", "v2", "v3"} {
		repo.Save(ModelMeta{
			Name:    "lr",
			Version: v,
			Type:    ModelTypeLR,
			CreatedAt: time.Now().Add(time.Duration(i) * time.Minute),
		}, data)
	}

	list := repo.List()
	if len(list) != 3 {
		t.Errorf("期望 3 条记录，得到 %d", len(list))
	}
	// 应按时间降序
	if list[0].Version != "v3" {
		t.Errorf("第一条应为最新版本 v3，得到 %s", list[0].Version)
	}
}

func TestRepo_ReadData(t *testing.T) {
	repo, _ := NewRepo(t.TempDir())
	payload := []byte(`{"test": 123}`)
	repo.Save(ModelMeta{Name: "lr", Version: "v1", Type: ModelTypeLR}, payload)

	got, err := repo.ReadData("v1")
	if err != nil {
		t.Fatalf("读取失败: %v", err)
	}
	if string(got) != string(payload) {
		t.Errorf("数据不匹配: %s vs %s", got, payload)
	}
}

func TestRepo_SaveVersionEmpty(t *testing.T) {
	repo, _ := NewRepo(t.TempDir())
	err := repo.Save(ModelMeta{Name: "lr", Type: ModelTypeLR}, []byte(`{}`))
	if err == nil {
		t.Error("期望版本为空时返回错误")
	}
}

func TestRepo_PersistIndex(t *testing.T) {
	dir := t.TempDir()
	repo1, _ := NewRepo(dir)
	repo1.Save(ModelMeta{Name: "lr", Version: "v1", Type: ModelTypeLR}, []byte(`{}`))

	// 重新加载仓库，索引应持久化
	repo2, err := NewRepo(dir)
	if err != nil {
		t.Fatalf("重新加载失败: %v", err)
	}
	_, ok := repo2.Get("v1")
	if !ok {
		t.Error("重新加载后应能找到 v1")
	}
}
